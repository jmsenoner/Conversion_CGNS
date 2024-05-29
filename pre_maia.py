from   mpi4py.MPI import COMM_WORLD as comm
import maia
import maia.pytree as PT
import numpy as np
import sys

def split_in_domains(dist_tree):
    """
    # The file contains several volumic "domains". CEDRE must have one mesh file per domain.
    """
    # Get domain names -> yiels "ROW(1)", ... "ROW(4)" for turbocel geometry
    domain_names = []
    for node in PT.get_nodes_from_label(dist_tree, "Family_t"):
        if PT.get_node_from_label(node, "FamilyBC_t") is None: domain_names.append(PT.get_name(node))

    # Create a dict "domainName => zones" -> adds block "domainxx" to domains "ROW(y)"
    domain2zones = {name:[] for name in domain_names}
    for zone in PT.get_nodes_from_label(dist_tree, "Zone_t"):
        family_name = PT.get_value(PT.get_node_from_label(zone, "FamilyName_t"))
        if family_name in domain2zones: domain2zones[family_name].append(zone)

    return domain2zones

def move_to_BC(parent,child):
    
    # Factorize displacement to NodeBC
    
    PT.add_child(parent, child)             # add mixing plane / periodicity to BCs
    PT.set_label(child, 'BC_t')             # relabel as BC
    PT.set_value(child, 'FamilySpecified')  # set value to FamilySpecifed
    
    return

dist_tree = maia.io.file_to_dist_tree('turbocel_7000.cgns', comm)

domain2zones = split_in_domains(dist_tree)

for (zonename, zones) in domain2zones.items(): # loop over zones / user domains
    tree = PT.new_CGNSTree()
    base = PT.new_CGNSBase('Base',parent=tree)
    for zone in zones:
        # Get nodes for boundary conditions (BCs) and grid connectivities
        node_zone_bc = PT.get_node_from_label(zone, "ZoneBC_t")
        node_zone_gc = PT.get_node_from_label(zone, "ZoneGridConnectivity_t")
        
        # Loop over nodes related to mixing plane BCs, which are stored as "GridConnectivity"
        for mp in PT.get_nodes_from_label(zone, "GridConnectivity_t"):
            move_to_BC(node_zone_bc, mp)
            PT.rm_child (node_zone_gc, mp) # node must be suppressed from connectivities for subsequent merge_zones to succeed
        
        # Nodes of BCs are renamed because the node name is then read by Cedre
        # For example, "BC_10_1" becomes "ROW(1)_INFLOW", much more readable to set BC types in Cedre
        for nodes_bc in PT.get_children(node_zone_bc):
            PT.set_name(nodes_bc, PT.get_value(PT.get_node_from_label(nodes_bc, "FamilyName_t")))
        PT.add_child(base, zone) # Modified blocs may now be added to the zone / user domain
    maia.algo.dist.convert_s_to_ngon(tree, comm)
    list_zones = {'Base/'+PT.get_name(zone):[] for zone in zones}
    maia.algo.dist.merge_zones(tree, list_zones, comm)

    PT.set_name(PT.get_node_from_label(tree, "Zone_t"), zonename) 
    
    # Move periodicities to ZoneBCs, similar to previous, but on the created tree
    node_tree_bc       = PT.get_node_from_label(tree, "ZoneBC_t")
    node_tree_gc       = PT.get_node_from_label(tree, "ZoneGridConnectivity_t")
    
    num_perio_bcs = len(PT.get_nodes_from_label(node_tree_gc, "GridConnectivity_t"))
    if num_perio_bcs % 2 != 0: 
        sys.exit("Only periodic BCs should remain at this stage")
    
    for perio_bcs in PT.get_nodes_from_label(node_tree_gc, "GridConnectivity_t"):
            move_to_BC(node_tree_bc, perio_bcs)
            # To be done, optional : shorten names of periodicites and relate to zone / user domain
    
    for nodes_bc in PT.get_children(node_tree_bc):
      family = PT.new_child(base, PT.get_name(nodes_bc), "Family_t")
      PT.new_child(family, "FamilyBC", "FamilyBC_t", "CedreBC")

    maia.io.dist_tree_to_file(tree, zonename.replace("(", "_").replace(")","_") + "ngon.cgns", comm)

