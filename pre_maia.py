from   mpi4py.MPI import COMM_WORLD as comm
import maia
import maia.pytree as PT

def split_in_domains(dist_tree):
    """
    The file contains several volumic "domains". CEDRE must have one mesh file per domain.
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

dist_tree = maia.io.file_to_dist_tree('turbocel_7000.cgns', comm)

domain2zones = split_in_domains(dist_tree)

for (zonename, zones) in domain2zones.items():
    tree = PT.new_CGNSTree()
    base = PT.new_CGNSBase('Base',parent=tree)
    for zone in zones:
        # il faudrait déplacer le noeud plan de melange de la gridConnectivity, Zone1 
        PT.add_child(base, zone)
    maia.algo.dist.convert_s_to_ngon(tree, comm)
    list_zones = {'Base/'+PT.get_name(zone):[] for zone in zones}
    # maia.algo.dist.merge_zones(tree, list_zones, comm)    # plante, a priori a cause des plans de mélange
    # maia.algo.dist.merge_connected_zones(tree, comm)      # plante
    maia.io.dist_tree_to_file(tree, zonename.replace("(", "_").replace(")","_") + "unst.cgns", comm)

