import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

class VASPXMLParser:
    def __init__(self, xml_path, output_dir=None):
        """
        Initialize VASP XML parser for any system
        
        Args:
            xml_path (str): Path to vasprun.xml file
            output_dir (str): Output directory for plots and summary
        """
        self.xml_path = xml_path
        self.output_dir = output_dir or os.path.join(os.path.dirname(xml_path), "vasp_analysis_nb_mos2")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize all data containers
        self.tree = None
        self.root = None
        self.system_info = {}
        self.energies = []
        self.positions_data = []
        self.forces_data = []
        self.stresses = []
        self.band_data = []
        self.dos_data = {}
        self.pdos_data = {}  # For projected DOS
        self.fermi_energy = np.nan
        self.atom_types = []
        self.n_atoms = 0
        self.atom_symbols = []
        self.bond_lengths = {}
        self.bond_angles = {}
        self.max_forces = []
        self.avg_forces = []
        self.band_gap = np.nan
        self.vbm_energy = np.nan  # Valence Band Maximum
        self.cbm_energy = np.nan  # Conduction Band Minimum
        self.magnetic_moments = []  # Magnetic moments data
        
    def safe_float(self, x, default=np.nan):
        """Safely convert to float with fallback"""
        try:
            return float(x) if x is not None else default
        except (ValueError, TypeError):
            return default
    
    def safe_int(self, x, default=0):
        """Safely convert to int with fallback"""
        try:
            return int(x) if x is not None else default
        except (ValueError, TypeError):
            return default
    
    def show_and_save(self, fig, filename):
        """Display and save figure"""
        try:
            plt.tight_layout()
            path = os.path.join(self.output_dir, filename)
            print(f"Saved: {filename}")
            plt.show()
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            plt.close(fig)

    def extract_magnetic_moments(self):
        """Extract magnetic moments data if available"""
        print("Extracting magnetic moments...")
        self.magnetic_moments = []
        
        for calc in self.root.findall(".//calculation"):
            # Try to find magnetic moments in different possible locations
            magmom_data = []
            
            # Method 1: Look for varray with name='magmom'
            magmom_array = calc.find("varray[@name='magmom']")
            if magmom_array is not None:
                for v in magmom_array.findall("v"):
                    if v.text:
                        try:
                            magmom = self.safe_float(v.text.strip())
                            magmom_data.append(magmom)
                        except (ValueError, TypeError):
                            continue
            
            # Method 2: Look for total magnetization in energy section
            if not magmom_data:
                energy_elem = calc.find("energy")
                if energy_elem is not None:
                    magmom_elem = energy_elem.find("i[@name='magmom']")
                    if magmom_elem is not None and magmom_elem.text:
                        try:
                            total_magmom = self.safe_float(magmom_elem.text.strip())
                            magmom_data = [total_magmom]  # Store as single value for total
                        except (ValueError, TypeError):
                            pass
            
            # Method 3: Look for magnetization in dos section
            if not magmom_data:
                dos_elem = calc.find("dos")
                if dos_elem is not None:
                    # Look for total magnetization
                    total_elem = dos_elem.find("total")
                    if total_elem is not None:
                        array_elem = total_elem.find("array")
                        if array_elem is not None:
                            # Look for magnetization data in array
                            set_elem = array_elem.find("set")
                            if set_elem is not None:
                                # This might contain spin-polarized DOS data
                                # For simplicity, we'll note that spin-polarized calculation was done
                                magmom_data = ["spin_polarized"]
            
            if magmom_data:
                self.magnetic_moments.append(magmom_data)
        
        print(f"Found magnetic moments for {len(self.magnetic_moments)} steps")
        if self.magnetic_moments:
            # Check if we have per-atom moments or total moments
            if isinstance(self.magnetic_moments[0], list) and len(self.magnetic_moments[0]) > 1:
                print(f"Per-atom magnetic moments available ({len(self.magnetic_moments[0])} atoms)")
            else:
                print("Total magnetic moment data available")

    def parse_xml(self):
        """Parse the VASP XML file"""
        try:
            print(f"Parsing {self.xml_path}...")
            self.tree = ET.parse(self.xml_path)
            self.root = self.tree.getroot()
            print("XML file loaded successfully")
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"File not found: {self.xml_path}")
            sys.exit(1)
    
    def extract_system_info(self):
        """Extract system information (atom types, numbers, etc.)"""
        print("Extracting system information...")
        
        # Get atom types and numbers
        atominfo = self.root.find(".//atominfo")
        if atominfo is not None:
            # Get number of atoms
            atoms_elem = atominfo.find("atoms")
            self.n_atoms = self.safe_int(atoms_elem.text) if atoms_elem is not None else 0
            
            # Get atom types
            types_elem = atominfo.find("types")
            n_types = self.safe_int(types_elem.text) if types_elem is not None else 0
            
            # Get atom type information
            types_array = atominfo.find("array[@name='atomtypes']")
            if types_array is not None:
                for set_elem in types_array.findall("set/rc"):
                    c_elements = set_elem.findall("c")
                    if len(c_elements) >= 2:
                        n_atoms_type = self.safe_int(c_elements[0].text)
                        symbol = c_elements[1].text.strip() if c_elements[1].text else "X"
                        self.atom_types.append((n_atoms_type, symbol))
            
            # Create atom symbols list
            for n_atoms_type, symbol in self.atom_types:
                self.atom_symbols.extend([symbol] * n_atoms_type)
        
        # Fallback if no atom info found
        if not self.atom_symbols and self.n_atoms > 0:
            self.atom_symbols = [f"Atom{i+1}" for i in range(self.n_atoms)]
        
        self.system_info = {
            'n_atoms': self.n_atoms,
            'n_types': len(self.atom_types),
            'atom_types': self.atom_types,
            'atom_symbols': self.atom_symbols
        }
        
        print(f"System: {self.n_atoms} atoms, {len(self.atom_types)} types")
        for n, symbol in self.atom_types:
            print(f"  {n} {symbol} atoms")
    
    def extract_energies(self):
        """Extract energies from all ionic steps"""
        print("Extracting energies...")
        self.energies = []
        
        for calc in self.root.findall(".//calculation"):
            energy_elem = calc.find("energy")
            if energy_elem is not None:
                # Try different energy types
                for energy_name in ['e_fr_energy', 'e_wo_entrp', 'e_0_energy']:
                    energy_val = energy_elem.find(f"i[@name='{energy_name}']")
                    if energy_val is not None and energy_val.text:
                        self.energies.append(self.safe_float(energy_val.text))
                        break
        
        print(f"Found {len(self.energies)} energy points")
    
    def extract_forces_positions(self):
        """Extract forces and positions from all steps"""
        print("Extracting positions and forces...")
        self.positions_data = []
        self.forces_data = []
        
        for calc in self.root.findall(".//calculation"):
            # Positions
            struct = calc.find("structure")
            if struct is not None:
                pos_array = struct.find("varray[@name='positions']")
                if pos_array is not None:
                    pos_step = []
                    for v in pos_array.findall("v"):
                        if v.text:
                            coords = [self.safe_float(x) for x in v.text.split()]
                            if len(coords) >= 3:
                                pos_step.append(coords[:3])
                    if pos_step:
                        self.positions_data.append(pos_step)
            
            # Forces
            forces_array = calc.find("varray[@name='forces']")
            if forces_array is not None:
                force_step = []
                for v in forces_array.findall("v"):
                    if v.text:
                        forces = [self.safe_float(x) for x in v.text.split()]
                        if len(forces) >= 3:
                            force_step.append(forces[:3])
                if force_step:
                    self.forces_data.append(force_step)
        
        print(f"Found {len(self.positions_data)} position steps")
        print(f"Found {len(self.forces_data)} force steps")
    
    def extract_stress(self):
        """Extract stress tensors"""
        print("Extracting stress data...")
        self.stresses = []
        
        for calc in self.root.findall(".//calculation"):
            stress_array = calc.find("varray[@name='stress']")
            if stress_array is not None:
                stress_step = []
                for v in stress_array.findall("v"):
                    if v.text:
                        stress_comp = [self.safe_float(x) for x in v.text.split()]
                        stress_step.extend(stress_comp)
                
                if len(stress_step) >= 9:  # Full stress tensor
                    self.stresses.append(stress_step[:9])
        
        print(f"Found {len(self.stresses)} stress tensors")
    
    def extract_electronic_structure(self):
        """Extract electronic structure data including DOS and projected DOS"""
        print("Extracting electronic structure...")
        
        self.band_data = []
        self.dos_data = {}
        self.pdos_data = {}
        fermi_candidates = []
        
        calculations = self.root.findall(".//calculation")
        if not calculations:
            print("No calculations found in XML")
            return
        
        last_calc = calculations[-1]
        
        # Extract Fermi energy first from DOS element or eigenvalue statistics
        fermi_elem = self.root.find(".//dos/i[@name='efermi']")
        if fermi_elem is not None and fermi_elem.text:
            self.fermi_energy = self.safe_float(fermi_elem.text)
            print(f"Found explicit Fermi energy: {self.fermi_energy:.6f} eV")
        
        # Extract band structure
        eigenval = last_calc.find(".//eigenvalues")
        if eigenval is not None:
            for array in eigenval.findall(".//array"):
                for set_elem in array.findall(".//set"):
                    for kpoint in set_elem.findall("set"):
                        for band in kpoint.findall("r"):
                            if band.text:
                                vals = band.text.split()
                                if len(vals) >= 2:
                                    energy = self.safe_float(vals[0])
                                    occupation = self.safe_float(vals[1])
                                    self.band_data.append({
                                        "energy": energy,
                                        "occupation": occupation
                                    })
                                    # Collect Fermi candidates if not already found
                                    if np.isnan(self.fermi_energy) and occupation > 0.5:
                                        fermi_candidates.append(energy)
        
        # If Fermi energy not found, infer from band data
        if np.isnan(self.fermi_energy) and fermi_candidates:
            self.fermi_energy = max(fermi_candidates)
            print(f"Inferred Fermi energy from band data: {self.fermi_energy:.6f} eV")
        
        # Extract DOS data
        dos_elem = last_calc.find("dos")
        if dos_elem is not None:
            print("Found DOS data")
            
            # Get total DOS
            total_dos = dos_elem.find("total/array/set/set")
            if total_dos is not None:
                energies = []
                dos_values = []
                for r in total_dos.findall("r"):
                    if r.text:
                        vals = r.text.split()
                        if len(vals) >= 2:
                            energies.append(self.safe_float(vals[0]))
                            dos_values.append(self.safe_float(vals[1]))
                
                if energies and dos_values:
                    self.dos_data['total'] = {
                        'energy': np.array(energies),
                        'dos': np.array(dos_values)
                    }
                    print(f"Total DOS: {len(energies)} points")
            
        # Extract DOS data
        dos_elem = last_calc.find("dos")
        if dos_elem is not None:
            print("Found DOS data")
            
            # Get total DOS
            total_dos = dos_elem.find("total/array/set/set")
            if total_dos is not None:
                energies = []
                dos_values = []
                for r in total_dos.findall("r"):
                    if r.text:
                        vals = r.text.split()
                        if len(vals) >= 2:
                            energies.append(self.safe_float(vals[0]))
                            dos_values.append(self.safe_float(vals[1]))
                
                if energies and dos_values:
                    self.dos_data['total'] = {
                        'energy': np.array(energies),
                        'dos': np.array(dos_values)
                    }
                    print(f"Total DOS: {len(energies)} points")
            
            # Extract partial DOS (projected DOS) if available - try multiple approaches
            self.extract_partial_dos_comprehensive(dos_elem)
        
        if np.isnan(self.fermi_energy):
            print("Warning: Fermi energy not found and could not be inferred")
    
    def extract_partial_dos_comprehensive(self, dos_elem):
        """Comprehensive PDOS extraction with multiple fallback methods"""
        print("Attempting comprehensive PDOS extraction...")
        
        # Method 1: Standard partial DOS extraction
        if self.extract_partial_dos(dos_elem):
            print("Successfully extracted PDOS using standard method")
            return True
        
        # Method 2: Try different XML structure patterns
        print("Trying alternative PDOS extraction methods...")
        
        # Look for different possible PDOS structures
        possible_paths = [
            "partial/array",
            "partial",
            ".//partial/array", 
            ".//partial"
        ]
        
        for path in possible_paths:
            partial_elem = dos_elem.find(path)
            if partial_elem is not None:
                print(f"Found partial DOS at path: {path}")
                if self.extract_partial_dos_alt_method(partial_elem):
                    return True
        
        # Method 3: Look for any array elements that might contain orbital data
        all_arrays = dos_elem.findall(".//array")
        print(f"Found {len(all_arrays)} array elements in DOS section")
        
        for i, array in enumerate(all_arrays):
            name_attr = array.get('name', '')
            print(f"  Array {i}: name='{name_attr}'")
            
            # Look for arrays that might contain orbital information
            if 'partial' in name_attr.lower() or 'orbital' in name_attr.lower():
                print(f"  Potential PDOS array found: {name_attr}")
                if self.extract_partial_dos_from_array(array):
                    return True
        
        print("No projected DOS data found in any attempted extraction method")
        return False
        
        if np.isnan(self.fermi_energy):
            print("Warning: Fermi energy not found and could not be inferred")
    
    def extract_partial_dos(self, dos_elem):
        """Extract partial DOS (projected DOS) for individual orbitals"""
        print("Extracting partial DOS (orbital projections)...")
        
        # Find partial DOS array
        partial_array = dos_elem.find("partial/array")
        if partial_array is None:
            print("No partial DOS data found")
            return
        
        # Parse orbital field names
        orbital_names = []
        for field in partial_array.findall("field"):
            orbital_name = field.text.strip() if field.text else ""
            if orbital_name and orbital_name.lower() != "energy":
                orbital_names.append(orbital_name)
        
        if not orbital_names:
            print("No orbital fields found in partial DOS")
            return
        
        print(f"Found orbital projections: {orbital_names}")
        
        # Find the data set
        set_elem = partial_array.find("set")
        if set_elem is None:
            print("No partial DOS data set found")
            return
        
        # Use energy from total DOS for consistency
        if 'total' in self.dos_data:
            energy_values = self.dos_data['total']['energy']
            print("Using energy values from total DOS for PDOS consistency")
        else:
            print("No total DOS available for energy reference")
            return
        
        # Initialize PDOS storage
        self.pdos_data['orbitals'] = orbital_names
        self.pdos_data['energy'] = energy_values
        self.pdos_data['atom_data'] = []
        
        # Parse data for each atom
        atom_count = 0
        for atom_set in set_elem.findall("set"):
            atom_data = {'orbitals': {}}
            
            # Initialize orbital arrays
            for orbital in orbital_names:
                atom_data['orbitals'][orbital] = []
            
            # Parse orbital data for this atom (skip energy column)
            for r_elem in atom_set.findall("r"):
                if r_elem.text:
                    values = r_elem.text.split()
                    # First column is energy, remaining are orbital projections
                    if len(values) >= len(orbital_names) + 1:
                        for i, orbital in enumerate(orbital_names):
                            orbital_idx = i + 1  # Skip energy column
                            if orbital_idx < len(values):
                                atom_data['orbitals'][orbital].append(self.safe_float(values[orbital_idx]))
            
            # Verify data completeness
            data_valid = True
            expected_points = len(energy_values)
            for orbital in orbital_names:
                if len(atom_data['orbitals'][orbital]) != expected_points:
                    print(f"Warning: Atom {atom_count} orbital {orbital} has {len(atom_data['orbitals'][orbital])} points, expected {expected_points}")
                    data_valid = False
                    break
            
            if data_valid:
                self.pdos_data['atom_data'].append(atom_data)
                atom_count += 1
        
        print(f"Successfully extracted partial DOS for {atom_count} atoms")
        
        # Create summed orbital projections across all atoms
        if self.pdos_data['atom_data'] and atom_count > 0:
            self.calculate_summed_pdos()
            return True
        else:
            print("Warning: No valid PDOS data extracted")
            return False
    
    def extract_partial_dos_alt_method(self, partial_elem):
        """Alternative method for PDOS extraction"""
        print("Trying alternative PDOS extraction method...")
        
        # Look for field definitions
        fields = partial_elem.findall(".//field")
        if not fields:
            return False
        
        orbital_names = []
        for field in fields:
            field_name = field.text.strip() if field.text else ""
            if field_name and field_name.lower() != "energy":
                orbital_names.append(field_name)
        
        if not orbital_names:
            print("No orbital fields found in alternative method")
            return False
        
        print(f"Alternative method found orbitals: {orbital_names}")
        
        # Look for data sets
        data_sets = partial_elem.findall(".//set")
        if not data_sets:
            return False
        
        # Use total DOS energy for consistency
        if 'total' not in self.dos_data:
            return False
        
        energy_values = self.dos_data['total']['energy']
        
        # Initialize storage
        self.pdos_data['orbitals'] = orbital_names
        self.pdos_data['energy'] = energy_values
        self.pdos_data['atom_data'] = []
        
        # Process data
        atom_count = 0
        for atom_set in data_sets:
            if atom_set.findall("set"):  # This is an atom container
                continue
            
            atom_data = {'orbitals': {orb: [] for orb in orbital_names}}
            
            for r_elem in atom_set.findall("r"):
                if r_elem.text:
                    values = r_elem.text.split()
                    if len(values) >= len(orbital_names) + 1:
                        for i, orbital in enumerate(orbital_names):
                            atom_data['orbitals'][orbital].append(self.safe_float(values[i + 1]))
            
            # Check data validity
            if all(len(atom_data['orbitals'][orb]) == len(energy_values) for orb in orbital_names):
                self.pdos_data['atom_data'].append(atom_data)
                atom_count += 1
        
        if atom_count > 0:
            print(f"Alternative method extracted PDOS for {atom_count} atoms")
            self.calculate_summed_pdos()
            return True
        
        return False
    
    def extract_partial_dos_from_array(self, array_elem):
        """Extract PDOS from a generic array element"""
        print("Trying to extract PDOS from array element...")
        
        # Look for field definitions
        fields = array_elem.findall("field")
        if not fields:
            return False
        
        orbital_names = []
        for field in fields:
            field_name = field.text.strip() if field.text else ""
            if field_name and field_name.lower() != "energy":
                orbital_names.append(field_name)
        
        if not orbital_names:
            return False
        
        # Look for data
        r_elements = array_elem.findall(".//r")
        if not r_elements:
            return False
        
        print(f"Array method found orbitals: {orbital_names}, {len(r_elements)} data points")
        
        # Simple extraction assuming single atom or summed data
        if 'total' not in self.dos_data:
            return False
        
        energy_values = self.dos_data['total']['energy']
        expected_points = len(energy_values)
        
        if len(r_elements) != expected_points:
            print(f"Data length mismatch: found {len(r_elements)}, expected {expected_points}")
            return False
        
        # Initialize storage
        self.pdos_data['orbitals'] = orbital_names
        self.pdos_data['energy'] = energy_values
        self.pdos_data['summed'] = {orb: [] for orb in orbital_names}
        
        # Extract data
        for r_elem in r_elements:
            if r_elem.text:
                values = r_elem.text.split()
                if len(values) >= len(orbital_names) + 1:
                    for i, orbital in enumerate(orbital_names):
                        self.pdos_data['summed'][orbital].append(self.safe_float(values[i + 1]))
        
        # Convert to numpy arrays
        for orbital in orbital_names:
            self.pdos_data['summed'][orbital] = np.array(self.pdos_data['summed'][orbital])
        
        # Create grouped orbitals
        self.create_orbital_groups_from_summed()
        
        print("Successfully extracted PDOS using array method")
        return True
    
    def create_orbital_groups_from_summed(self):
        """Create orbital groups from already summed data"""
        if 'summed' not in self.pdos_data:
            return
        
        orbital_names = self.pdos_data['orbitals']
        self.pdos_data['grouped'] = {}
        
        # Group orbitals by type
        s_orbitals = [orb for orb in orbital_names if orb.strip().lower() in ['s']]
        if s_orbitals:
            self.pdos_data['grouped']['s'] = sum([self.pdos_data['summed'][orb] for orb in s_orbitals])
        
        p_orbitals = [orb for orb in orbital_names if orb.strip().lower() in ['py', 'pz', 'px']]
        if p_orbitals:
            self.pdos_data['grouped']['p'] = sum([self.pdos_data['summed'][orb] for orb in p_orbitals])
        
        d_orbitals = [orb for orb in orbital_names if orb.strip().lower() in ['dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2', 'dz^2', 'dx2']]
        if d_orbitals:
            self.pdos_data['grouped']['d'] = sum([self.pdos_data['summed'][orb] for orb in d_orbitals])
        
        f_orbitals = [orb for orb in orbital_names if orb.strip().lower().startswith('f')]
        if f_orbitals:
            self.pdos_data['grouped']['f'] = sum([self.pdos_data['summed'][orb] for orb in f_orbitals])
        
        print(f"Created orbital groups from summed data: {list(self.pdos_data['grouped'].keys())}")
    
    def calculate_summed_pdos(self):
        """Calculate summed PDOS across all atoms for each orbital type"""
        print("Calculating summed orbital projections...")
        
        orbital_names = self.pdos_data['orbitals']
        n_points = len(self.pdos_data['energy'])
        
        # Initialize summed orbitals
        self.pdos_data['summed'] = {}
        for orbital in orbital_names:
            self.pdos_data['summed'][orbital] = np.zeros(n_points)
        
        # Sum over all atoms
        for atom_idx, atom_data in enumerate(self.pdos_data['atom_data']):
            for orbital in orbital_names:
                if orbital in atom_data['orbitals'] and len(atom_data['orbitals'][orbital]) == n_points:
                    self.pdos_data['summed'][orbital] += np.array(atom_data['orbitals'][orbital])
                else:
                    print(f"Warning: Atom {atom_idx} orbital {orbital} has inconsistent data length")
        
        # Group orbitals by type (s, p, d, f) - more comprehensive grouping
        self.pdos_data['grouped'] = {}
        
        # s orbitals (various naming conventions)
        s_orbitals = [orb for orb in orbital_names if orb.strip().lower() in ['s']]
        if s_orbitals:
            self.pdos_data['grouped']['s'] = sum([self.pdos_data['summed'][orb] for orb in s_orbitals])
            print(f"Found s orbitals: {s_orbitals}")
        
        # p orbitals
        p_orbitals = [orb for orb in orbital_names if orb.strip().lower() in ['py', 'pz', 'px']]
        if p_orbitals:
            self.pdos_data['grouped']['p'] = sum([self.pdos_data['summed'][orb] for orb in p_orbitals])
            print(f"Found p orbitals: {p_orbitals}")
        
        # d orbitals
        d_orbitals = [orb for orb in orbital_names if orb.strip().lower() in ['dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2', 'dz^2', 'dx2']]
        if d_orbitals:
            self.pdos_data['grouped']['d'] = sum([self.pdos_data['summed'][orb] for orb in d_orbitals])
            print(f"Found d orbitals: {d_orbitals}")
        
        # f orbitals
        f_orbitals = [orb for orb in orbital_names if orb.strip().lower().startswith('f')]
        if f_orbitals:
            self.pdos_data['grouped']['f'] = sum([self.pdos_data['summed'][orb] for orb in f_orbitals])
            print(f"Found f orbitals: {f_orbitals}")
        
        print(f"Summed PDOS calculated for orbital groups: {list(self.pdos_data['grouped'].keys())}")
        
        # Debug: Print some statistics
        for orbital_type, pdos_values in self.pdos_data['grouped'].items():
            max_val = np.max(pdos_values)
            print(f"  {orbital_type}: max DOS = {max_val:} states/eV")
    
    def calculate_derived_quantities(self):
        """Calculate derived quantities like bond lengths, bond angles, force magnitudes, and band gap"""
        print("Calculating derived quantities...")
        
        # Force magnitudes per step
        self.max_forces = []
        self.avg_forces = []
        
        for step_forces in self.forces_data:
            if step_forces:
                magnitudes = [np.linalg.norm(f) for f in step_forces]
                self.max_forces.append(max(magnitudes))
                self.avg_forces.append(np.mean(magnitudes))
        
        # Bond lengths for small systems (< 10 atoms)
        self.bond_lengths = {}
        self.bond_angles = {}
        
        if self.n_atoms <= 10 and self.positions_data:
            print("Calculating bond lengths...")
            for i, j in combinations(range(self.n_atoms), 2):
                bond_name = f"{self.atom_symbols[i]}{i+1}-{self.atom_symbols[j]}{j+1}"
                self.bond_lengths[bond_name] = []
                
                for step_pos in self.positions_data:
                    if len(step_pos) > max(i, j):
                        pos1 = np.array(step_pos[i])
                        pos2 = np.array(step_pos[j])
                        distance = np.linalg.norm(pos1 - pos2)
                        self.bond_lengths[bond_name].append(distance)
            
            # Calculate bond angles (for systems with 3+ atoms)
            if self.n_atoms >= 3:
                print("Calculating bond angles...")
                for center_idx in range(self.n_atoms):
                    other_atoms = [i for i in range(self.n_atoms) if i != center_idx]
                    for i, j in combinations(other_atoms, 2):
                        angle_name = f"{self.atom_symbols[i]}{i+1}-{self.atom_symbols[center_idx]}{center_idx+1}-{self.atom_symbols[j]}{j+1}"
                        self.bond_angles[angle_name] = []
                        
                        for step_pos in self.positions_data:
                            if len(step_pos) > max(i, j, center_idx):
                                pos_center = np.array(step_pos[center_idx])
                                pos_i = np.array(step_pos[i])
                                pos_j = np.array(step_pos[j])
                                vec1 = pos_i - pos_center
                                vec2 = pos_j - pos_center
                                norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                                if norm1 > 1e-10 and norm2 > 1e-10:
                                    cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
                                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                                    angle = np.degrees(np.arccos(cos_angle))
                                    self.bond_angles[angle_name].append(angle)
        
        # Band gap calculation using inferred Fermi energy
        self.band_gap = np.nan
        self.vbm_energy = np.nan
        self.cbm_energy = np.nan
        
        if self.band_data:
            occupied = [b['energy'] for b in self.band_data if b['occupation'] > 0.5]
            unoccupied = [b['energy'] for b in self.band_data if b['occupation'] <= 0.5]
            
            if occupied and unoccupied:
                self.vbm_energy = max(occupied)
                self.cbm_energy = min(unoccupied)
                self.band_gap = self.cbm_energy - self.vbm_energy
                print(f"Calculated band gap: {self.band_gap:.6f} eV")
                print(f"VBM energy: {self.vbm_energy:.6f} eV")
                print(f"CBM energy: {self.cbm_energy:.6f} eV")
                print(f"VBM relative to Fermi: {self.vbm_energy - self.fermi_energy:.6f} eV")
                print(f"CBM relative to Fermi: {self.cbm_energy - self.fermi_energy:.6f} eV")

    def plot_energy_convergence(self):
        """Plot energy convergence with proper scaling"""
        if not self.energies:
            print("No energy data to plot")
            return
        
        # Energy vs steps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        steps = np.array(range(1, len(self.energies) + 1))
        energies_array = np.array(self.energies)
        
        # Plot 1: Energy values
        ax1.plot(steps, energies_array, 'bo-', markersize=4, linewidth=1.5)
        ax1.set_xlabel("Ionic Step")
        ax1.set_ylabel("Energy (eV)")
        ax1.set_title("Energy Convergence")
        ax1.grid(True, alpha=0.3)
        
        # Set proper axis limits
        if len(steps) > 1:
            ax1.set_xlim(0.5, len(steps) + 0.5)
        
        if len(self.energies) > 1:
            energy_range = np.ptp(energies_array)  # peak-to-peak
            if energy_range > 1e-12:  # Non-zero range
                center = np.mean(energies_array)
                margin = energy_range * 0.15
                ax1.set_ylim(center - energy_range/2 - margin, center + energy_range/2 + margin)
            else:
                # For constant energy
                ax1.set_ylim(energies_array[0] - 0.001, energies_array[0] + 0.001)
        
        # Plot 2: Energy differences
        if len(self.energies) > 1:
            energy_diffs = np.abs(np.diff(energies_array))
            diff_steps = steps[1:]
            
            # Only plot if differences are meaningful
            if np.max(energy_diffs) > 1e-12:
                ax2.semilogy(diff_steps, energy_diffs, 'ro-', 
                            markersize=4, linewidth=1.5)
                ax2.set_xlabel("Ionic Step")
                ax2.set_ylabel("|ΔE| (eV)")
                ax2.set_title("Energy Difference")
                ax2.grid(True, alpha=0.3)
                
                # Set proper axis limits
                ax2.set_xlim(1.5, len(steps) + 0.5)
                min_diff = np.min(energy_diffs[energy_diffs > 0]) if np.any(energy_diffs > 0) else 1e-12
                max_diff = np.max(energy_diffs)
                ax2.set_ylim(min_diff * 0.1, max_diff * 10)
            else:
                ax2.text(0.5, 0.5, 'Energy converged\n(differences < 1e-12 eV)', 
                        transform=ax2.transAxes, ha='center', va='center', fontsize=12)
                ax2.set_xlabel("Ionic Step")
                ax2.set_ylabel("|ΔE| (eV)")
                ax2.set_title("Energy Difference")
        
        self.show_and_save(fig, "energy_convergence.png")
    
    def plot_forces_convergence(self):
        """Plot force convergence with proper scaling"""
        if not self.max_forces:
            print("No force data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = np.array(range(1, len(self.max_forces) + 1))
        max_forces_array = np.array(self.max_forces)
        
        ax.plot(steps, max_forces_array, 'ro-', label='Max Force', markersize=4, linewidth=1.5)
        
        if self.avg_forces:
            avg_forces_array = np.array(self.avg_forces)
            ax.plot(steps, avg_forces_array, 'bo-', label='Avg Force', markersize=4, linewidth=1.5)
        
        # Use log scale only if the force range is significant
        force_range = np.ptp(max_forces_array)
        min_force = np.min(max_forces_array)
        
        if force_range > 0 and min_force > 0 and (np.max(max_forces_array) / min_force) > 10:
            ax.set_yscale("log")
            ax.set_ylim(min_force * 0.5, np.max(max_forces_array) * 2)
        else:
            # Use linear scale for small variations
            if force_range > 1e-12:
                margin = force_range * 0.1
                ax.set_ylim(min_force - margin, np.max(max_forces_array) + margin)
        
        ax.set_xlim(0.5, len(steps) + 0.5)
        ax.set_xlabel("Ionic Step")
        ax.set_ylabel("Force (eV/Å)")
        ax.set_title("Force Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.show_and_save(fig, "forces_convergence.png")
    
    def plot_band_structure(self):
        """Plot electronic band structure with proper labeling of VBM, CBM, and Fermi level"""
        if not self.band_data:
            print("No band data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        energies = np.array([b['energy'] for b in self.band_data])
        energies_shifted = energies - self.fermi_energy
        occupations = np.array([b['occupation'] for b in self.band_data])
        
        # Create band indices
        band_indices = np.arange(len(energies))
        
        # Separate occupied and unoccupied bands
        occupied_mask = occupations > 0.5
        unoccupied_mask = occupations <= 0.5
        
        # Plot occupied and unoccupied bands with different colors
        if np.any(occupied_mask):
            ax.scatter(band_indices[occupied_mask], energies_shifted[occupied_mask], 
                      c='red', alpha=0.7, s=20, 
                      label=f'Occupied Bands ({np.sum(occupied_mask)})', 
                      edgecolors='darkred', linewidth=0.5, zorder=3)
        
        if np.any(unoccupied_mask):
            ax.scatter(band_indices[unoccupied_mask], energies_shifted[unoccupied_mask], 
                      c='blue', alpha=0.7, s=20, 
                      label=f'Unoccupied Bands ({np.sum(unoccupied_mask)})', 
                      edgecolors='darkblue', linewidth=0.5, zorder=3)
        
        # Add Fermi level if available
        if not np.isnan(self.fermi_energy):
            ax.axhline(0, color='green', linestyle='--', linewidth=3,
                      label=f'Fermi Level (0 eV)', zorder=2)
            
            # Add shaded region around Fermi level for clarity
            ax.axhspan(-0.1, 0.1, color='green', alpha=0.1, zorder=1)
        
        # Mark VBM and CBM if available
        if not np.isnan(self.vbm_energy) and not np.isnan(self.cbm_energy):
            vbm_shifted = self.vbm_energy - self.fermi_energy
            cbm_shifted = self.cbm_energy - self.fermi_energy
            
            # Find band indices closest to VBM and CBM
            vbm_idx = np.argmin(np.abs(energies_shifted[occupied_mask] - vbm_shifted))
            cbm_idx = np.argmin(np.abs(energies_shifted[unoccupied_mask] - cbm_shifted))
            
            # Get actual positions in the full arrays
            vbm_positions = band_indices[occupied_mask][vbm_idx]
            cbm_positions = band_indices[unoccupied_mask][cbm_idx]
            
            # Mark VBM with a special marker
            ax.scatter([vbm_positions], [vbm_shifted], 
                      c='gold', s=50, marker='o', edgecolors='black', linewidth=1.5,
                      label=f'VBM: {vbm_shifted:.3f} eV', zorder=4)
            
            # Mark CBM with a special marker  
            ax.scatter([cbm_positions], [cbm_shifted],
                      c='lime', s=50, marker='o', edgecolors='black', linewidth=1.5,
                      label=f'CBM: {cbm_shifted:.3f} eV', zorder=4)
            
            # Add text annotations with energy values
            ax.annotate(f'VBM: {vbm_shifted:.3f} eV', 
                       xy=(vbm_positions, vbm_shifted), 
                       xytext=(10, 20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                       zorder=5)
            
            ax.annotate(f'CBM: {cbm_shifted:.3f} eV', 
                       xy=(cbm_positions, cbm_shifted), 
                       xytext=(10, -30), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lime', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                       zorder=5)
        
        # Set proper axis limits
        ax.set_xlim(-0.5, len(energies) - 0.5)
        
        if len(energies) > 0:
            # Focus on energy range around Fermi level if available
            if not np.isnan(self.fermi_energy):
                # Show ±10 eV around Fermi level, but adjust if data range is smaller
                energy_range = np.ptp(energies_shifted)
                if energy_range < 40:  # If total range is less than
                    center = 0
                    half_range = max(10, energy_range * 0.6)
                    ax.set_ylim(center - half_range, center + half_range)
                else:
                    ax.set_ylim(-15, 15)  
            else:
                # No Fermi level, use full energy range with margin
                energy_range = np.ptp(energies_shifted)
                margin = energy_range * 0.1
                ax.set_ylim(np.min(energies_shifted) - margin, np.max(energies_shifted) + margin)
        
        # Add band gap information if available
        if not np.isnan(self.band_gap):
            gap_text = f'Band Gap: {self.band_gap:.3f} eV'
            if self.band_gap < 0.1:
                gap_text += '\n(Metallic or very small gap)'
            
            ax.text(0.02, 0.98, gap_text, 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                   fontsize=12, zorder=5)
        
        ax.set_xlabel("Band Index")
        ax.set_ylabel("Energy - E$_F$ (eV)")
        ax.set_title("Electronic Band Structure\n(VBM: Valence Band Maximum, CBM: Conduction Band Minimum)")
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        self.show_and_save(fig, "electronic_band_structure.png")
    
    def plot_magnetic_moments(self):
        """Plot magnetic moments evolution if available"""
        if not self.magnetic_moments:
            print("No magnetic moment data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = np.array(range(1, len(self.magnetic_moments) + 1))
        
        # Check if we have per-atom data or total data
        first_moment = self.magnetic_moments[0]
        
        if isinstance(first_moment, list) and len(first_moment) > 1:
            # Per-atom magnetic moments
            magnetic_array = np.array(self.magnetic_moments)
            
            # Plot individual atom moments
            for atom_idx in range(magnetic_array.shape[1]):
                symbol = self.atom_symbols[atom_idx] if atom_idx < len(self.atom_symbols) else f"Atom{atom_idx+1}"
                ax.plot(steps, magnetic_array[:, atom_idx], 
                       marker='o', markersize=4, linewidth=1.5, 
                       label=f'{symbol}{atom_idx+1}')
            
            ax.set_ylabel("Magnetic Moment per Atom (μB)")
            title_suffix = "Per-Atom"
            
            # Also plot total magnetic moment
            total_moments = np.sum(magnetic_array, axis=1)
            ax.plot(steps, total_moments, 'k-', linewidth=3, marker='s', 
                   markersize=6, label='Total Magnetic Moment', alpha=0.8)
            
        else:
            # Total magnetic moment
            if isinstance(first_moment, list):
                # Single value in list
                moments = [m[0] if isinstance(m, list) and len(m) == 1 else m for m in self.magnetic_moments]
            else:
                moments = self.magnetic_moments
            
            # Convert to numeric values, skipping non-numeric entries
            numeric_moments = []
            for m in moments:
                if isinstance(m, (int, float)):
                    numeric_moments.append(float(m))
                elif isinstance(m, list) and len(m) == 1 and isinstance(m[0], (int, float)):
                    numeric_moments.append(float(m[0]))
                else:
                    numeric_moments.append(0.0)  # Default for non-numeric
            
            ax.plot(steps, numeric_moments, 'ro-', markersize=6, linewidth=2)
            ax.set_ylabel("Total Magnetic Moment (μB)")
            title_suffix = "Total"
        
        # Set proper axis limits
        ax.set_xlim(0.5, len(steps) + 0.5)
        
        # Use appropriate y-axis scaling
        if len(self.magnetic_moments) > 1:
            all_moments_flat = []
            for moment in self.magnetic_moments:
                if isinstance(moment, list):
                    all_moments_flat.extend(moment)
                else:
                    all_moments_flat.append(moment)
            
            # Filter out non-numeric values
            numeric_only = [m for m in all_moments_flat if isinstance(m, (int, float))]
            if numeric_only:
                moment_range = np.ptp(numeric_only)
                min_moment = np.min(numeric_only)
                max_moment = np.max(numeric_only)
                
                if moment_range > 1e-10:
                    margin = moment_range * 0.15
                    ax.set_ylim(min_moment - margin, max_moment + margin)
                else:
                    # Nearly constant moments
                    center = (min_moment + max_moment) / 2
                    ax.set_ylim(center - 0.1, center + 0.1)
        
        ax.set_xlabel("Ionic Step")
        ax.set_title(f"Magnetic Moment Evolution ({title_suffix})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        self.show_and_save(fig, "magnetic_moments.png")
        
        # Save magnetic moment data to file
        self.save_magnetic_moment_data()
    
    def save_magnetic_moment_data(self):
        """Save magnetic moment data to text file"""
        if not self.magnetic_moments:
            return
        
        magmom_file_path = os.path.join(self.output_dir, "magnetic_moments.txt")
        
        with open(magmom_file_path, "w") as f:
            f.write("Magnetic Moments Data\n")
            f.write("=" * 40 + "\n")
            
            # Check data type
            first_moment = self.magnetic_moments[0]
            
            if isinstance(first_moment, list) and len(first_moment) > 1:
                # Per-atom data
                f.write("Per-atom magnetic moments (μB)\n")
                f.write("-" * 40 + "\n")
                
                # Write header
                f.write(f"{'Step':<8}")
                for atom_idx in range(len(first_moment)):
                    symbol = self.atom_symbols[atom_idx] if atom_idx < len(self.atom_symbols) else f"Atom{atom_idx+1}"
                    f.write(f"{symbol+str(atom_idx+1):<12}")
                f.write(f"{'Total':<12}\n")
                
                # Write data
                for step_idx, moments in enumerate(self.magnetic_moments):
                    f.write(f"{step_idx+1:<8}")
                    total = 0.0
                    for moment in moments:
                        if isinstance(moment, (int, float)):
                            f.write(f"{moment:<12.6f}")
                            total += moment
                        else:
                            f.write(f"{'N/A':<12}")
                    f.write(f"{total:<12.6f}\n")
            else:
                # Total magnetic moment data
                f.write("Total magnetic moments (micro B)\n")
                f.write("-" * 40 + "\n")
                f.write("Step    Magnetic_Moment\n")
                
                for step_idx, moment in enumerate(self.magnetic_moments):
                    if isinstance(moment, list) and len(moment) == 1:
                        moment_val = moment[0]
                    else:
                        moment_val = moment
                    
                    if isinstance(moment_val, (int, float)):
                        f.write(f"{step_idx+1:<8}{moment_val:.6f}\n")
                    else:
                        f.write(f"{step_idx+1:<8}N/A\n")
        
        print(f"Magnetic moment data saved to: magnetic_moments.txt")
    
    def plot_density_of_states(self):
        """Plot density of states (DOS) with proper Fermi level alignment and orbital projections"""
        if 'total' not in self.dos_data:
            print("No DOS data to plot")
            return
        
        dos_info = self.dos_data['total']
        energies = dos_info['energy']
        dos_values = dos_info['dos']
        
        # Check if we have PDOS data with better debugging
        has_pdos = 'grouped' in self.pdos_data and self.pdos_data['grouped']
        print(f"PDOS data available: {has_pdos}")
        if has_pdos:
            print(f"PDOS orbital groups: {list(self.pdos_data['grouped'].keys())}")
            for orbital_type, pdos_vals in self.pdos_data['grouped'].items():
                print(f"  {orbital_type}: {len(pdos_vals)} points, max = {np.max(pdos_vals):}")
        else:
            print("PDOS debug info:")
            print(f"  'grouped' in self.pdos_data: {'grouped' in self.pdos_data}")
            if 'grouped' in self.pdos_data:
                print(f"  self.pdos_data['grouped']: {self.pdos_data['grouped']}")
            print(f"  'summed' in self.pdos_data: {'summed' in self.pdos_data}")
            print(f"  'orbitals' in self.pdos_data: {'orbitals' in self.pdos_data}")
        
        # Create figure with subplots
        if has_pdos:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = None
        
        # CRITICAL FIX: Shift energies relative to Fermi level for proper alignment
        if not np.isnan(self.fermi_energy):
            energies_shifted = energies - self.fermi_energy
            print(f"DOS energies shifted by Fermi level: {self.fermi_energy:.6f} eV")
        else:
            energies_shifted = energies
            print("Warning: Fermi energy not available, plotting absolute energies")
        
        # Plot total DOS on left subplot
        ax1.plot(dos_values, energies_shifted, 'k-', linewidth=2, label='Total DOS')
        
        # Add Fermi level line at 0 eV (CRITICAL: This should align with gap in DOS)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, 
                  label='Fermi Level (0 eV)')
        
        # Fill occupied states (below Fermi level)
        occupied_mask = energies_shifted <= 0
        if np.any(occupied_mask):
            ax1.fill_betweenx(energies_shifted[occupied_mask], 0, 
                           dos_values[occupied_mask], 
                           alpha=0.3, color='red', label='Occupied States')
        
        # Fill unoccupied states (above Fermi level)
        unoccupied_mask = energies_shifted > 0
        if np.any(unoccupied_mask):
            ax1.fill_betweenx(energies_shifted[unoccupied_mask], 0, 
                           dos_values[unoccupied_mask], 
                           alpha=0.3, color='blue', label='Unoccupied States')
        
        # Set proper axis limits for total DOS
        dos_max = np.max(dos_values) * 1.05
        ax1.set_xlim(-dos_max * 0.05, dos_max)  # Small negative space for clarity
        
        # Set energy axis to focus on relevant range around Fermi level
        self.set_dos_y_limits(ax1, energies_shifted)
        
        ax1.set_xlabel("Density of States (states/eV)")
        ax1.set_ylabel("Energy - E$_F$ (eV)")
        ax1.set_title("Total Density of States")
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Add band gap information if available
        if not np.isnan(self.band_gap):
            ax1.text(0.98, 0.98, f'Band Gap: {self.band_gap:.3f} eV', 
                   transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot projected DOS if available
        if has_pdos and ax2 is not None:
            print("Plotting projected DOS...")
            self.plot_projected_dos(ax2, energies_shifted)
        elif ax2 is not None:
            # Show message on right subplot if no PDOS data
            ax2.text(0.5, 0.5, 'No Projected DOS\nData Available', 
                    transform=ax2.transAxes, ha='center', va='center', 
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax2.set_title("Projected DOS (Not Available)")
            ax2.set_xlabel("Projected DOS (states/eV)")
            ax2.set_ylabel("Energy - E$_F$ (eV)")
        
        self.show_and_save(fig, "density_of_states.png")
        
        # Save DOS data to file
        self.save_dos_data()
        
        # Create separate detailed PDOS plot if available
        if has_pdos:
            print("Creating detailed PDOS plot...")
            self.plot_detailed_pdos(energies_shifted)
        else:
            print("Skipping detailed PDOS plot - no data available")
    
    def plot_projected_dos(self, ax, energies_shifted):
        """Plot projected density of states for orbital groups"""
        colors = {'s': 'red', 'p': 'blue', 'd': 'green', 'f': 'orange'}
        labels = {'s': 's orbital', 'p': 'p orbitals', 'd': 'd orbitals', 'f': 'f orbitals'}
        
        max_pdos = 0
        for orbital_type, pdos_values in self.pdos_data['grouped'].items():
            if orbital_type in colors and len(pdos_values) == len(energies_shifted):
                ax.plot(pdos_values, energies_shifted, 
                       color=colors[orbital_type], linewidth=2, 
                       label=labels.get(orbital_type, orbital_type))
                max_pdos = max(max_pdos, np.max(pdos_values))
        
        # Add Fermi level line at 0 eV
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2, 
                  label='Fermi Level (0 eV)')
        
        # Set proper axis limits
        ax.set_xlim(-max_pdos * 0.05, max_pdos * 1.05)
        self.set_dos_y_limits(ax, energies_shifted)
        
        ax.set_xlabel("Projected DOS (states/eV)")
        ax.set_ylabel("Energy - E$_F$ (eV)")
        ax.set_title("Orbital Projected Density of States")
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    def plot_detailed_pdos(self, energies_shifted):
        """Create detailed plot of individual orbital contributions"""
        if 'summed' not in self.pdos_data:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colors for different orbitals
        orbital_colors = {
            's': 'red', 's ': 'red',
            'px': 'blue', 'py': 'green', 'pz': 'orange',
            'px ': 'blue', 'py ': 'green', 'pz ': 'orange',
            'dxy': 'purple', 'dyz': 'brown', 'dz2': 'pink', 
            'dxz': 'gray', 'dx2-y2': 'olive',
            'dxy ': 'purple', 'dyz ': 'brown', 'dz2 ': 'pink', 
            'dxz ': 'gray', 'dx2-y2 ': 'olive'
        }
        
        max_pdos = 0
        plotted_orbitals = []
        
        # Plot individual orbitals
        for orbital, pdos_values in self.pdos_data['summed'].items():
            if len(pdos_values) == len(energies_shifted):
                color = orbital_colors.get(orbital, 'black')
                ax.plot(pdos_values, energies_shifted, 
                       color=color, linewidth=1.5, 
                       label=orbital.strip(), alpha=0.8)
                max_pdos = max(max_pdos, np.max(pdos_values))
                plotted_orbitals.append(orbital)
        
        # Add Fermi level line at 0 eV
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2, 
                  label='Fermi Level (0 eV)')
        
        # Set proper axis limits
        ax.set_xlim(-max_pdos * 0.05, max_pdos * 1.05)
        self.set_dos_y_limits(ax, energies_shifted)
        
        ax.set_xlabel("Projected DOS (states/eV)")
        ax.set_ylabel("Energy - E$_F$ (eV)")
        ax.set_title("Detailed Orbital Projected Density of States")
        
        # Handle legend based on number of orbitals
        if len(plotted_orbitals) <= 8:
            ax.legend(loc='best', framealpha=0.9)
        elif len(plotted_orbitals) <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
        else:
            # Too many orbitals, show summary in text box instead
            ax.text(0.98, 0.02, f'{len(plotted_orbitals)} orbitals plotted', 
                   transform=ax.transAxes, verticalalignment='bottom', 
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.grid(True, alpha=0.3)
        
        self.show_and_save(fig, "detailed_orbital_pdos.png")
    
    def set_dos_y_limits(self, ax, energies_shifted):
        """Set proper Y-axis limits for DOS plots focused on relevant energy range"""
        if len(energies_shifted) > 1:
            # Default to reasonable range around Fermi level
            energy_range = np.ptp(energies_shifted)
            
            # If we have a large energy range, focus on ±15 eV around Fermi level
            if energy_range > 30:
                ax.set_ylim(-15, 15)
            elif energy_range > 20:
                ax.set_ylim(-12, 12)
            else:
                # For smaller ranges, show the full range with some margin
                center = np.mean(energies_shifted)
                margin = max(1.0, energy_range * 0.1)  # At least 1 eV margin
                ax.set_ylim(center - energy_range/2 - margin, center + energy_range/2 + margin)
        else:
            # Default range if we don't have enough data
            ax.set_ylim(-10, 10)
    
    def save_dos_data(self):
        """Save DOS data to text file"""
        if 'total' not in self.dos_data:
            return
        
        dos_info = self.dos_data['total']
        energies = dos_info['energy']
        dos_values = dos_info['dos']
        
        dos_file_path = os.path.join(self.output_dir, "dos_data.txt")
        
        with open(dos_file_path, "w") as f:
            f.write("Density of States Data\n")
            f.write("=" * 40 + "\n")
            f.write(f"Fermi Energy: {self.fermi_energy:.6f} eV\n")
            f.write(f"Number of points: {len(energies)}\n")
            f.write("\n")
            f.write("Energy (eV)    Energy-EF (eV)    DOS (states/eV)\n")
            f.write("-" * 50 + "\n")
            
            for i, (energy, dos) in enumerate(zip(energies, dos_values)):
                energy_shifted = energy - self.fermi_energy if not np.isnan(self.fermi_energy) else np.nan
                f.write(f"{energy:12.6f}    {energy_shifted:12.6f}    {dos:12.6f}\n")
        
        # Save PDOS data if available
        if 'summed' in self.pdos_data and 'energy' in self.pdos_data:
            pdos_file_path = os.path.join(self.output_dir, "pdos_data.txt")
            with open(pdos_file_path, "w") as f:
                f.write("Projected Density of States Data\n")
                f.write("=" * 50 + "\n")
                f.write(f"Fermi Energy: {self.fermi_energy:.6f} eV\n")
                f.write(f"Orbitals: {', '.join(self.pdos_data['orbitals'])}\n")
                f.write("\n")
                
                # Write header
                header = "Energy (eV)    Energy-EF (eV)    " + "    ".join([f"{orb.strip():<12}" for orb in self.pdos_data['orbitals']])
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                
                # Write data
                n_points = len(self.pdos_data['energy'])
                for i in range(n_points):
                    energy = self.pdos_data['energy'][i]
                    energy_shifted = energy - self.fermi_energy if not np.isnan(self.fermi_energy) else np.nan
                    line = f"{energy:12.6f}    {energy_shifted:12.6f}    "
                    for orbital in self.pdos_data['orbitals']:
                        if i < len(self.pdos_data['summed'][orbital]):
                            line += f"{self.pdos_data['summed'][orbital][i]:12.6f}    "
                    f.write(line + "\n")
            
            print(f"PDOS data saved to: pdos_data.txt")
        
        print(f"DOS data saved to: dos_data.txt")
    
    def plot_stress_evolution(self):
        """Plot stress tensor evolution with proper scaling"""
        if not self.stresses:
            print("No stress data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        stress_array = np.array(self.stresses)
        steps = np.array(range(1, len(self.stresses) + 1))
        
        # Plot diagonal stress components (most important)
        stress_labels = ['XX', 'YY', 'ZZ', 'XY', 'YZ', 'XZ']
        diagonal_indices = [0, 4, 8]  # XX, YY, ZZ positions in flattened tensor
        colors = ['red', 'blue', 'green']
        
        plotted_any = False
        for i, idx in enumerate(diagonal_indices):
            if stress_array.shape[1] > idx:
                ax.plot(steps, stress_array[:, idx], 
                       color=colors[i], label=f'σ_{stress_labels[i]}', 
                       marker='o', markersize=4, linewidth=1.5)
                plotted_any = True
        
        if not plotted_any:
            print("No valid stress components to plot")
            plt.close(fig)
            return
        
        # Set proper axis limits
        ax.set_xlim(0.5, len(steps) + 0.5)
        
        if stress_array.size > 0:
            # Get range of diagonal stress components only
            diagonal_stresses = stress_array[:, diagonal_indices]
            min_stress = np.min(diagonal_stresses)
            max_stress = np.max(diagonal_stresses)
            stress_range = max_stress - min_stress
            
            if stress_range > 1e-10:  # Non-zero range
                margin = stress_range * 0.15
                ax.set_ylim(min_stress - margin, max_stress + margin)
            else:
                # Nearly constant stress
                center = (min_stress + max_stress) / 2
                ax.set_ylim(center - 1, center + 1)
        
        ax.set_xlabel("Ionic Step")
        ax.set_ylabel("Stress (kBar)")
        ax.set_title("Stress Evolution (Diagonal Components)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.show_and_save(fig, "stress_evolution.png")
    
    def plot_atomic_positions(self):
        """Plot 3D atomic positions evolution with proper scaling"""
        if not self.positions_data or self.n_atoms == 0:
            print("No position data to plot")
            return
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color map for different atoms
        colors = plt.cm.tab10(np.linspace(0, 1, min(self.n_atoms, 10)))
        if self.n_atoms > 10:
            colors = plt.cm.viridis(np.linspace(0, 1, self.n_atoms))
        
        # Collect all positions for proper scaling
        all_positions = []
        trajectories = []
        
        # Plot trajectory for each atom
        for atom_idx in range(min(self.n_atoms, len(self.positions_data[0]))):
            trajectory = np.array([step[atom_idx] for step in self.positions_data 
                                 if len(step) > atom_idx])
            
            if len(trajectory) > 0:
                trajectories.append(trajectory)
                all_positions.extend(trajectory.tolist())
                
                symbol = self.atom_symbols[atom_idx] if atom_idx < len(self.atom_symbols) else f"Atom{atom_idx+1}"
                color = colors[atom_idx % len(colors)]
                
                # Plot trajectory
                if len(trajectory) > 1:
                    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                           color=color, alpha=0.7, linewidth=2, 
                           label=f'{symbol}{atom_idx+1}')
                
                # Mark initial position (circle)
                ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                          color=color, s=120, marker='o', edgecolors='black', linewidth=1)
                
                # Mark final position (square) if different from initial
                if len(trajectory) > 1:
                    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                              color=color, s=120, marker='s', edgecolors='black', linewidth=1)
        
        # Set equal aspect ratio and proper limits
        if all_positions:
            all_positions = np.array(all_positions)
            
            # Calculate ranges for each dimension
            ranges = []
            centers = []
            for i in range(3):
                coord_range = np.ptp(all_positions[:, i])
                center = np.mean(all_positions[:, i])
                ranges.append(coord_range)
                centers.append(center)
            
            # Use the maximum range to ensure equal scaling
            max_range = max(ranges) if ranges else 1.0
            padding = max(max_range * 0.1, 0.1)  # At least 0.1 Å padding
            
            # Set limits with equal scaling
            ax.set_xlim(centers[0] - max_range/2 - padding, centers[0] + max_range/2 + padding)
            ax.set_ylim(centers[1] - max_range/2 - padding, centers[1] + max_range/2 + padding)
            ax.set_zlim(centers[2] - max_range/2 - padding, centers[2] + max_range/2 + padding)
        
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        ax.set_title("Atomic Trajectory Evolution\n(Circles: initial, Squares: final)")
        
        # Only show legend if reasonable number of atoms
        if self.n_atoms <= 15:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Add text box with atom count instead
            ax.text2D(0.02, 0.98, f'{self.n_atoms} atoms', transform=ax.transAxes, 
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        self.show_and_save(fig, "atomic_trajectories_3d.png")
    
    def plot_bond_lengths_and_angles(self):
        """Plot bond lengths and angles evolution for small systems with proper scaling"""
        if not self.bond_lengths or self.n_atoms > 10:
            print("No bond length/angle data to plot (system too large or no data)")
            return
        
        # Create subplot layout based on available data
        has_angles = bool(self.bond_angles)
        
        if has_angles:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        else:
            fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Determine number of steps from first bond
        first_bond_distances = next(iter(self.bond_lengths.values()))
        steps = np.array(range(1, len(first_bond_distances) + 1))
        
        # Plot bond lengths
        bond_items = list(self.bond_lengths.items())[:10]  # Limit to 10 bonds
        colors = plt.cm.tab10(np.linspace(0, 1, len(bond_items)))
        
        for i, (bond_name, distances) in enumerate(bond_items):
            distances_array = np.array(distances)
            ax1.plot(steps, distances_array, color=colors[i], marker='o', 
                    markersize=4, label=bond_name, linewidth=1.5)
        
        # Set proper axis limits for bond lengths
        ax1.set_xlim(0.5, len(steps) + 0.5)
        
        if bond_items:
            all_distances = np.array([d for _, dist_list in bond_items for d in dist_list])
            if len(all_distances) > 0:
                min_dist = np.min(all_distances)
                max_dist = np.max(all_distances)
                dist_range = max_dist - min_dist
                
                if dist_range > 1e-10:
                    margin = max(dist_range * 0.1, 0.01)  # At least 0.01 Å margin
                    ax1.set_ylim(min_dist - margin, max_dist + margin)
                else:
                    # Nearly constant distances
                    center = (min_dist + max_dist) / 2
                    ax1.set_ylim(center - 0.05, center + 0.05)
        
        ax1.set_xlabel("Ionic Step")
        ax1.set_ylabel("Bond Length (Å)")
        ax1.set_title("Bond Length Evolution")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot bond angles if available
        if has_angles:
            angle_items = list(self.bond_angles.items())[:10]  # Limit to 10 angles
            angle_colors = plt.cm.Set3(np.linspace(0, 1, len(angle_items)))
            
            for i, (angle_name, angles) in enumerate(angle_items):
                if len(angles) == len(steps):  # Ensure same length as steps
                    angles_array = np.array(angles)
                    ax2.plot(steps, angles_array, color=angle_colors[i], marker='s', 
                            markersize=4, label=angle_name, linewidth=1.5)
            
            # Set proper axis limits for bond angles
            ax2.set_xlim(0.5, len(steps) + 0.5)
            
            if angle_items:
                all_angles = np.array([a for _, angle_list in angle_items for a in angle_list])
                if len(all_angles) > 0:
                    min_angle = np.min(all_angles)
                    max_angle = np.max(all_angles)
                    angle_range = max_angle - min_angle
                    
                    if angle_range > 1e-10:
                        margin = max(angle_range * 0.1, 1.0)  # At least 1 degree margin
                        ax2.set_ylim(min_angle - margin, max_angle + margin)
                    else:
                        # Nearly constant angles
                        center = (min_angle + max_angle) / 2
                        ax2.set_ylim(center - 5, center + 5)
            
            ax2.set_xlabel("Ionic Step")
            ax2.set_ylabel("Bond Angle (°)")
            ax2.set_title("Bond Angle Evolution")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        self.show_and_save(fig, "bond_lengths_and_angles.png")
        
        # Save detailed bond data to file
        self.save_bond_data()
    
    def save_bond_data(self):
        """Save detailed bond lengths and angles data to file"""
        if not self.bond_lengths and not self.bond_angles:
            return
        
        bond_data_path = os.path.join(self.output_dir, "bond_angles_lengths.txt")
        
        with open(bond_data_path, "w") as f:
            f.write("Bond Lengths and Angles Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            # Write bond lengths
            if self.bond_lengths:
                f.write("BOND LENGTHS (Å):\n")
                f.write("-" * 20 + "\n")
                
                # Header
                steps = range(1, len(next(iter(self.bond_lengths.values()))) + 1)
                f.write(f"{'Bond':<15}")
                for step in steps:
                    f.write(f"{'Step'+str(step):<10}")
                f.write(f"{'Average':<10}{'StdDev':<10}{'Final':<10}\n")
                
                # Data
                for bond_name, distances in self.bond_lengths.items():
                    distances_array = np.array(distances)
                    f.write(f"{bond_name:<15}")
                    for dist in distances:
                        f.write(f"{dist:<10.4f}")
                    f.write(f"{np.mean(distances_array):<10.4f}")
                    f.write(f"{np.std(distances_array):<10.4f}")
                    f.write(f"{distances[-1]:<10.4f}\n")
                f.write("\n")
            
            # Write bond angles
            if self.bond_angles:
                f.write("BOND ANGLES (°):\n")
                f.write("-" * 20 + "\n")
                
                # Header
                if self.bond_angles:
                    steps = range(1, len(next(iter(self.bond_angles.values()))) + 1)
                    f.write(f"{'Angle':<20}")
                    for step in steps:
                        f.write(f"{'Step'+str(step):<10}")
                    f.write(f"{'Average':<10}{'StdDev':<10}{'Final':<10}\n")
                    
                    # Data
                    for angle_name, angles in self.bond_angles.items():
                        angles_array = np.array(angles)
                        f.write(f"{angle_name:<20}")
                        for angle in angles:
                            f.write(f"{angle:<10.2f}")
                        f.write(f"{np.mean(angles_array):<10.2f}")
                        f.write(f"{np.std(angles_array):<10.2f}")
                        f.write(f"{angles[-1]:<10.2f}\n")
        
        print(f"Bond data saved to: bond_angles_lengths.txt")
    
    def generate_summary(self):
        """Generate and save analysis summary"""
        print("Generating summary...")
        
        summary_lines = [
            "VASP XML Analysis Summary",
            "=" * 50,
            f"System Information:",
            f"  Total atoms: {self.n_atoms}",
            f"  Atom types: {len(self.atom_types)}",
        ]
        
        for n, symbol in self.atom_types:
            summary_lines.append(f"    {n} {symbol} atoms")
        
        summary_lines.extend([
            "",
            f"Calculation Steps:",
            f"  Ionic steps: {len(self.energies)}",
            f"  Position steps: {len(self.positions_data)}",
            f"  Force steps: {len(self.forces_data)}",
            f"  Stress steps: {len(self.stresses)}",
            "",
            f"Final Results:",
        ])
        
        if self.energies:
            summary_lines.append(f"  Final energy: {self.energies[-1]:.6f} eV")
            if len(self.energies) > 1:
                energy_change = abs(self.energies[-1] - self.energies[0])
                summary_lines.append(f"  Total energy change: {energy_change:.6f} eV")
                # Convergence check
                if len(self.energies) >= 2:
                    final_energy_change = abs(self.energies[-1] - self.energies[-2])
                    summary_lines.append(f"  Final step energy change: {final_energy_change:.8f} eV")
        
        if self.max_forces:
            summary_lines.append(f"  Final max force: {self.max_forces[-1]:.6f} eV/A")
            if len(self.max_forces) > 1:
                summary_lines.append(f"  Initial max force: {self.max_forces[0]:.6f} eV/A")
        
        # Electronic properties section
        summary_lines.append("")
        summary_lines.append("Electronic Properties:")
        
        if not np.isnan(self.fermi_energy):
            summary_lines.append(f"  Fermi energy: {self.fermi_energy:.6f} eV")
        else:
            summary_lines.append(f"  Fermi energy: Not available")
        
        if not np.isnan(self.band_gap):
            if self.band_gap < 0.1:
                summary_lines.append(f"  Band gap: {self.band_gap:.6f} eV (metallic or very small gap)")
            else:
                summary_lines.append(f"  Band gap: {self.band_gap:.6f} eV")
            
            if not np.isnan(self.vbm_energy) and not np.isnan(self.cbm_energy):
                summary_lines.append(f"  VBM energy: {self.vbm_energy:.6f} eV")
                summary_lines.append(f"  CBM energy: {self.cbm_energy:.6f} eV")
                summary_lines.append(f"  VBM relative to Fermi: {self.vbm_energy - self.fermi_energy:.6f} eV")
                summary_lines.append(f"  CBM relative to Fermi: {self.cbm_energy - self.fermi_energy:.6f} eV")
        else:
            summary_lines.append(f"  Band gap: Not calculated")
        
        if self.band_data:
            occupied = sum(1 for b in self.band_data if b['occupation'] > 0.5)
            summary_lines.append(f"  Occupied bands: {occupied}")
            summary_lines.append(f"  Total bands: {len(self.band_data)}")
        
        if 'total' in self.dos_data:
            dos_points = len(self.dos_data['total']['energy'])
            summary_lines.append(f"  DOS points: {dos_points}")
        
        if 'orbitals' in self.pdos_data:
            summary_lines.append(f"  Projected DOS orbitals: {', '.join(self.pdos_data['orbitals'])}")
        
        # Magnetic properties section
        if self.magnetic_moments:
            summary_lines.append("")
            summary_lines.append("Magnetic Properties:")
            summary_lines.append(f"  Magnetic moments available for {len(self.magnetic_moments)} steps")
            
            first_moment = self.magnetic_moments[0]
            if isinstance(first_moment, list) and len(first_moment) > 1:
                summary_lines.append(f"  Per-atom magnetic moments available")
                # Calculate final total magnetic moment
                final_moments = self.magnetic_moments[-1]
                if all(isinstance(m, (int, float)) for m in final_moments):
                    total_final_moment = sum(final_moments)
                    summary_lines.append(f"  Final total magnetic moment: {total_final_moment:.6f} μB")
            else:
                summary_lines.append(f"  Total magnetic moments available")
                if isinstance(first_moment, (int, float)) or (isinstance(first_moment, list) and len(first_moment) == 1):
                    final_moment = self.magnetic_moments[-1]
                    if isinstance(final_moment, list) and len(final_moment) == 1:
                        final_moment = final_moment[0]
                    summary_lines.append(f"  Final magnetic moment: {final_moment} micro B")
        
        # Structural properties section
        if self.bond_lengths or self.bond_angles:
            summary_lines.append("")
            summary_lines.append("Structural Properties:")
            
            if self.bond_lengths:
                summary_lines.append(f"  Number of bonds analyzed: {len(self.bond_lengths)}")
                summary_lines.append("  Final bond lengths:")
                for bond_name, distances in self.bond_lengths.items():
                    if distances:
                        summary_lines.append(f"    {bond_name}: {distances[-1]:} A")
            
            if self.bond_angles:
                summary_lines.append(f"  Number of angles analyzed: {len(self.bond_angles)}")
                summary_lines.append("  Final bond angles:")
                for angle_name, angles in self.bond_angles.items():
                    if angles:
                        summary_lines.append(f"    {angle_name}: {angles[-1]:.2f}°")
        
        # Add convergence assessment
        summary_lines.append("")
        summary_lines.append("Convergence Assessment:")
        
        if len(self.energies) > 1:
            final_energy_change = abs(self.energies[-1] - self.energies[-2])
            if final_energy_change < 1e-5:
                summary_lines.append(f"  Energy: CONVERGED (|E| = {final_energy_change:.2e} eV)")
            elif final_energy_change < 1e-3:
                summary_lines.append(f"  Energy: Nearly converged (|E| = {final_energy_change:.2e} eV)")
            else:
                summary_lines.append(f"  Energy: NOT CONVERGED (|E| = {final_energy_change:.2e} eV)")
        
        if self.max_forces:
            final_max_force = self.max_forces[-1]
            if final_max_force < 0.01:
                summary_lines.append(f"  Forces: CONVERGED (max = {final_max_force:} eV/A)")
            elif final_max_force < 0.05:
                summary_lines.append(f"  Forces: Nearly converged (max = {final_max_force:} eV/A)")
            else:
                summary_lines.append(f"  Forces: NOT CONVERGED (max = {final_max_force:} eV/A)")
        
        summary_text = "\n".join(summary_lines)
        
        # Save summary
        summary_path = os.path.join(self.output_dir, "analysis_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary_text)
        
        print("\n" + summary_text)
        print(f"\nAnalysis complete! Results saved in: {self.output_dir}")
        
        return summary_text
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting VASP XML analysis...")
        
        # Parse and extract data
        self.parse_xml()
        self.extract_system_info()
        self.extract_energies()
        self.extract_forces_positions()
        self.extract_stress()
        self.extract_electronic_structure()
        self.extract_magnetic_moments()  # NEW: Extract magnetic moments
        self.calculate_derived_quantities()
        
        # Generate all plots
        print("\nGenerating plots...")
        self.plot_energy_convergence()
        self.plot_forces_convergence()
        self.plot_band_structure()  # UPDATED: Now includes VBM/CBM labeling
        self.plot_density_of_states()  # Includes PDOS plots
        self.plot_stress_evolution()
        self.plot_atomic_positions()
        self.plot_bond_lengths_and_angles()
        self.plot_magnetic_moments()  # NEW: Plot magnetic moments
        
        # Generate summary
        self.generate_summary()


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Analyze VASP XML output files")
    parser.add_argument("xml_file", help="Path to vasprun.xml file")
    parser.add_argument("-o", "--output", help="Output directory for results")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.xml_file):
        print(f"Error: File {args.xml_file} not found!")
        sys.exit(1)
    
    # Initialize and run analysis
    analyzer = VASPXMLParser(args.xml_file, args.output)
    
    try:
        analyzer.run_full_analysis()
    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # If run as script, use command line arguments
    if len(sys.argv) > 1:
        main()
    else:
        xml_path = "snsband_vasprun.xml"
        analyzer = VASPXMLParser(xml_path)
        analyzer.run_full_analysis()