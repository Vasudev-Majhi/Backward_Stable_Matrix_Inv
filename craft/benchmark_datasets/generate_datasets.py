"""
Generate benchmark edge list files for the compiled-transformer solver paper.
Run once: python generate_datasets.py
"""
import os

base = "craft/benchmark_datasets"
os.makedirs(base, exist_ok=True)

# =============================================================
# DATASET 1: Zachary's Karate Club (34 nodes, 78 edges)
# Social network - friendships in a university karate club
# Source: Zachary 1977, via NetworkX karate_club_graph()
# =============================================================
karate_edges = [
    (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),(0,12),(0,13),
    (0,17),(0,19),(0,21),(0,31),
    (1,2),(1,3),(1,7),(1,13),(1,17),(1,19),(1,21),(1,30),
    (2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),(2,32),
    (3,7),(3,12),(3,13),
    (4,6),(4,10),
    (5,6),(5,10),(5,16),
    (6,16),
    (8,30),(8,32),(8,33),
    (9,33),
    (13,33),
    (14,32),(14,33),
    (15,32),(15,33),
    (18,32),(18,33),
    (19,33),
    (20,32),(20,33),
    (22,32),(22,33),
    (23,25),(23,27),(23,29),(23,32),(23,33),
    (24,25),(24,27),(24,31),
    (25,31),
    (26,29),(26,33),
    (27,33),
    (28,31),(28,33),
    (29,32),(29,33),
    (30,32),(30,33),
    (31,32),(31,33),
    (32,33),
]
with open(f"{base}/karate.edges", "w") as f:
    f.write("# Zachary's Karate Club Social Network\n")
    f.write("# 34 nodes, 78 edges, unit weight (all = 1.0)\n")
    f.write("# Domain: Social Network\n")
    f.write("# Source: W.W. Zachary, J. Anthropol. Res. 33, 452-473 (1977)\n")
    f.write("# Available via NetworkX: networkx.generators.social.karate_club_graph()\n")
    f.write("# Format: node_i node_j weight\n")
    for u, v in karate_edges:
        f.write(f"{u} {v} 1.0\n")
print(f"Karate: {len(karate_edges)} edges, {max(max(u,v) for u,v in karate_edges)+1} nodes")


# =============================================================
# DATASET 2: Dolphins Social Network (62 nodes, 159 edges)
# Undirected social network of bottlenose dolphins off NZ
# Source: Lusseau et al. Behav. Ecol. Sociobiol. 54 (2003)
# Edge list from GML (RishujeetRai/DolphinsSocialNetworkAnalysis)
# =============================================================
dolphins_raw = [
    (8,3),(9,5),(9,6),(10,0),(10,2),(13,5),(13,6),(13,9),(14,0),(14,3),
    (15,0),(16,14),(17,1),(17,6),(17,9),(17,13),(18,15),(19,1),(19,7),(20,8),
    (20,16),(20,18),(21,18),(22,17),(24,14),(24,15),(24,18),(25,17),(26,1),(26,25),
    (27,1),(27,7),(27,17),(27,25),(27,26),(28,1),(28,8),(28,20),(29,10),(29,18),
    (29,21),(29,24),(30,7),(30,19),(30,28),(31,17),(32,9),(32,13),(33,12),(33,14),
    (33,16),(33,21),(34,14),(34,33),(35,29),(36,1),(36,20),(36,23),(37,8),(37,14),
    (37,16),(37,21),(37,33),(37,34),(37,36),(38,14),(38,16),(38,20),(38,33),(39,36),
    (40,0),(40,7),(40,14),(40,15),(40,33),(40,36),(40,37),(41,1),(41,9),(41,13),
    (42,0),(42,2),(42,10),(42,30),(43,14),(43,29),(43,33),(43,37),(43,38),(44,2),
    (44,20),(44,34),(44,38),(45,8),(45,15),(45,18),(45,21),(45,23),(45,24),(45,29),
    (45,37),(46,43),(47,0),(47,10),(47,20),(47,28),(47,30),(47,42),(49,34),(49,46),
    (50,14),(50,16),(50,20),(50,33),(50,42),(50,45),(51,4),(51,11),(51,18),(51,21),
    (51,23),(51,24),(51,29),(51,45),(51,50),(52,14),(52,29),(52,38),(52,40),(53,43),
    (54,1),(54,6),(54,7),(54,13),(54,19),(54,41),(55,15),(55,51),(56,5),(56,6),
    (57,5),(57,6),(57,9),(57,13),(57,17),(57,39),(57,41),(57,48),(57,54),(58,38),
    (59,3),(59,8),(59,15),(59,36),(59,45),(60,32),(61,2),(61,37),(61,53),
]
with open(f"{base}/dolphins.edges", "w") as f:
    f.write("# Dolphins Social Network\n")
    f.write("# 62 nodes (IDs 0-61), 159 edges, unit weight (all = 1.0)\n")
    f.write("# Domain: Biological / Animal Social Network\n")
    f.write("# Source: Lusseau et al. Behav. Ecol. Sociobiol. 54, 396-405 (2003)\n")
    f.write("# Available: https://github.com/RishujeetRai/DolphinsSocialNetworkAnalysis\n")
    f.write("#            SuiteSparse Newman/dolphins\n")
    f.write("# Format: node_i node_j weight\n")
    for u, v in dolphins_raw:
        f.write(f"{u} {v} 1.0\n")
nodes_d = set()
for u, v in dolphins_raw:
    nodes_d.add(u); nodes_d.add(v)
print(f"Dolphins: {len(dolphins_raw)} edges, {len(nodes_d)} distinct nodes, max_id={max(nodes_d)}")


# =============================================================
# DATASET 3: Les Miserables Co-appearance Network
# 77 nodes, 254 edges, WEIGHTED (co-appearance count)
# Source: Knuth 1993 (Stanford GraphBase)
# JSON from vega/vega-datasets miserables.json
# =============================================================
lesmis_nodes = {
    0:"Myriel", 1:"Napoleon", 2:"Mlle.Baptistine", 3:"Mme.Magloire", 4:"CountessdeLo",
    5:"Geborand", 6:"Champtercier", 7:"Cravatte", 8:"Count", 9:"OldMan", 10:"Labarre",
    11:"Valjean", 12:"Marguerite", 13:"Mme.deR", 14:"Isabeau", 15:"Gervais",
    16:"Tholomyes", 17:"Listolier", 18:"Fameuil", 19:"Blacheville", 20:"Favourite",
    21:"Dahlia", 22:"Zephine", 23:"Fantine", 24:"Mme.Thenardier", 25:"Thenardier",
    26:"Cosette", 27:"Javert", 28:"Fauchelevent", 29:"Bamatabois", 30:"Perpetue",
    31:"Simplice", 32:"Scaufflaire", 33:"Woman1", 34:"Judge", 35:"Champmathieu",
    36:"Brevet", 37:"Chenildieu", 38:"Cochepaille", 39:"Pontmercy", 40:"Boulatruelle",
    41:"Eponine", 42:"Anzelma", 43:"Woman2", 44:"MotherInnocent", 45:"Gribier",
    46:"Jondrette", 47:"Mme.Burgon", 48:"Gavroche", 49:"Gillenormand", 50:"Magnon",
    51:"Mlle.Gillenormand", 52:"Mme.Pontmercy", 53:"Mlle.Vaubois", 54:"Lt.Gillenormand",
    55:"Marius", 56:"BaronessT", 57:"Mabeuf", 58:"Enjolras", 59:"Combeferre",
    60:"Prouvaire", 61:"Feuilly", 62:"Courfeyrac", 63:"Bahorel", 64:"Bossuet",
    65:"Joly", 66:"Grantaire", 67:"MotherPlutarch", 68:"Gueulemer", 69:"Babet",
    70:"Claquesous", 71:"Montparnasse", 72:"Toussaint", 73:"Child1", 74:"Child2",
    75:"Brujon", 76:"Mme.Hucheloup"
}
lesmis_edges = [
    (1,0,1),(2,0,8),(3,0,10),(3,2,6),(4,0,1),(5,0,1),(6,0,1),(7,0,1),(8,0,2),(9,0,1),
    (11,10,1),(11,3,3),(11,2,3),(11,0,5),(12,11,1),(13,11,1),(14,11,1),(15,11,1),
    (17,16,4),(18,16,4),(18,17,4),(19,16,4),(19,17,4),(19,18,4),(20,16,3),(20,17,3),
    (20,18,3),(20,19,4),(21,16,3),(21,17,3),(21,18,3),(21,19,3),(21,20,5),(22,16,3),
    (22,17,3),(22,18,3),(22,19,3),(22,20,4),(22,21,4),(23,16,3),(23,17,3),(23,18,3),
    (23,19,3),(23,20,4),(23,21,4),(23,22,4),(23,12,2),(23,11,9),(24,23,2),(24,11,7),
    (25,24,13),(25,23,1),(25,11,12),(26,24,4),(26,11,31),(26,16,1),(26,25,1),
    (27,11,17),(27,23,5),(27,25,5),(27,24,1),(27,26,1),(28,11,8),(28,27,1),
    (29,23,1),(29,27,1),(29,11,2),(30,23,1),(31,30,2),(31,11,3),(31,23,2),(31,27,1),
    (32,11,1),(33,11,2),(33,27,1),(34,11,3),(34,29,2),(35,11,3),(35,34,3),(35,29,2),
    (36,34,2),(36,35,2),(36,11,2),(36,29,1),(37,34,2),(37,35,2),(37,36,2),(37,11,2),
    (37,29,1),(38,34,2),(38,35,2),(38,36,2),(38,37,2),(38,11,2),(38,29,1),
    (39,25,1),(40,25,1),(41,24,2),(41,25,3),(42,41,2),(42,25,2),(42,24,1),(43,11,3),
    (43,26,1),(43,27,1),(44,28,3),(44,11,1),(45,28,2),(47,46,1),(48,47,2),(48,25,1),
    (48,27,1),(48,11,1),(49,26,3),(49,11,2),(50,49,1),(50,24,1),(51,49,9),(51,26,2),
    (51,11,2),(52,51,1),(52,39,1),(53,51,1),(54,51,2),(54,49,1),(54,26,1),(55,51,6),
    (55,49,12),(55,39,1),(55,54,1),(55,26,21),(55,11,19),(55,16,1),(55,25,2),
    (55,41,5),(55,48,4),(56,49,1),(56,55,1),(57,55,1),(57,41,1),(57,48,1),(58,55,7),
    (58,48,7),(58,27,6),(58,57,1),(58,11,4),(59,58,15),(59,55,5),(59,48,6),(59,57,2),
    (60,48,1),(60,58,4),(60,59,2),(61,48,2),(61,58,6),(61,60,2),(61,59,5),(61,57,1),
    (61,55,1),(62,55,9),(62,58,17),(62,59,13),(62,48,7),(62,57,2),(62,41,1),(62,61,6),
    (62,60,3),(63,59,5),(63,48,5),(63,62,6),(63,57,2),(63,58,4),(63,61,3),(63,60,2),
    (63,55,1),(64,55,5),(64,62,12),(64,48,5),(64,63,4),(64,58,10),(64,61,6),(64,60,2),
    (64,59,9),(64,57,1),(64,11,1),(65,63,5),(65,64,7),(65,48,3),(65,62,5),(65,58,5),
    (65,61,5),(65,60,2),(65,59,5),(65,57,1),(65,55,2),(66,64,3),(66,58,3),(66,59,1),
    (66,62,2),(66,65,2),(66,48,1),(66,63,1),(66,61,1),(66,60,1),(67,57,3),(68,25,5),
    (68,11,1),(68,24,1),(68,27,1),(68,48,1),(68,41,1),(69,25,6),(69,68,6),(69,11,1),
    (69,24,1),(69,27,2),(69,48,1),(69,41,1),(70,25,4),(70,69,4),(70,68,4),(70,11,1),
    (70,24,1),(70,27,1),(70,41,1),(70,58,1),(71,27,1),(71,69,2),(71,68,2),(71,70,2),
    (71,11,1),(71,48,1),(71,41,1),(71,25,1),(72,26,2),(72,27,1),(72,11,1),(73,48,2),
    (74,48,2),(74,73,3),(75,69,3),(75,68,3),(75,25,3),(75,48,1),(75,41,1),(75,70,1),
    (75,71,1),(76,64,1),(76,65,1),(76,66,1),(76,63,1),(76,62,1),(76,48,1),(76,58,1),
]
with open(f"{base}/lesmis.edges", "w") as f:
    f.write("# Les Miserables Character Co-appearance Network\n")
    f.write("# 77 nodes, 254 edges, WEIGHTED (weight = co-appearance count = conductance)\n")
    f.write("# Domain: Literary / Humanities Network\n")
    f.write("# Source: D.E. Knuth, The Stanford GraphBase (1993)\n")
    f.write("# JSON: https://raw.githubusercontent.com/vega/vega-datasets/master/data/miserables.json\n")
    f.write("# Format: node_i node_j weight\n")
    for u, v, w in lesmis_edges:
        f.write(f"{u} {v} {w}\n")
with open(f"{base}/lesmis_nodes.txt", "w") as f:
    f.write("# Les Miserables node index -> character name\n")
    for idx, name in sorted(lesmis_nodes.items()):
        f.write(f"{idx} {name}\n")
print(f"Les Mis: {len(lesmis_edges)} edges, {len(lesmis_nodes)} nodes")


# =============================================================
# DATASET 4: Political Books (polbooks) Network
# 105 nodes, 441 edges
# Books about US politics co-purchased on Amazon (2004 election)
# Source: V. Krebs; edge list via melaniewalsh/sample-social-network-datasets
# =============================================================
polbooks_raw_lines = """1,0,1
2,0,1
3,0,1
3,1,1
4,0,1
4,2,1
5,0,1
5,1,1
5,2,1
5,3,1
5,4,1
6,0,1
6,1,1
6,4,1
6,5,1
7,2,1
7,5,1
7,6,1
8,3,1
9,3,1
9,8,1
10,3,1
10,6,1
10,8,1
11,3,1
11,8,1
11,9,1
11,10,1
12,3,1
12,6,1
12,8,1
12,9,1
12,10,1
12,11,1
13,3,1
13,8,1
13,11,1
13,12,1
14,3,1
14,8,1
14,9,1
14,11,1
14,12,1
14,7,1
15,3,1
15,10,1
15,12,1
16,3,1
16,10,1
16,15,1
17,3,1
17,11,1
17,12,1
17,13,1
18,3,1
18,6,1
18,12,1
19,3,1
19,10,1
20,3,1
20,8,1
20,9,1
20,11,1
21,3,1
21,8,1
21,10,1
21,11,1
22,3,1
22,6,1
22,8,1
22,11,1
23,3,1
23,8,1
23,12,1
23,21,1
24,3,1
24,8,1
24,9,1
24,12,1
24,20,1
25,3,1
25,6,1
25,14,1
25,22,1
26,3,1
26,8,1
26,11,1
26,14,1
26,24,1
27,3,1
27,8,1
27,9,1
27,11,1
27,23,1
28,4,1
29,4,1
29,6,1
29,11,1
29,13,1
30,4,1
30,7,1
31,4,1
31,30,1
32,8,1
32,12,1
32,13,1
32,23,1
33,32,1
33,8,1
33,10,1
33,12,1
33,23,1
35,34,1
35,8,1
35,10,1
36,34,1
36,35,1
36,12,1
37,34,1
37,8,1
37,10,1
37,35,1
37,33,1
38,34,1
38,10,1
38,35,1
38,12,1
38,37,1
38,33,1
39,34,1
39,10,1
39,35,1
39,12,1
39,38,1
39,33,1
40,8,1
40,35,1
40,12,1
40,13,1
40,20,1
40,22,1
40,39,1
40,24,1
40,25,1
40,26,1
40,27,1
41,8,1
41,9,1
41,40,1
41,12,1
41,36,1
41,27,1
42,8,1
42,40,1
42,13,1
42,39,1
43,8,1
43,35,1
43,13,1
43,42,1
44,8,1
44,40,1
44,35,1
44,12,1
44,13,1
45,8,1
45,9,1
45,40,1
45,11,1
45,26,1
46,8,1
46,12,1
47,9,1
47,40,1
47,41,1
47,11,1
47,12,1
47,13,1
47,42,1
47,36,1
47,37,1
47,17,1
47,33,1
47,45,1
47,46,1
47,23,1
47,24,1
47,26,1
47,27,1
48,9,1
48,20,1
49,9,1
49,20,1
49,31,1
49,48,1
50,9,1
50,11,1
51,9,1
52,9,1
52,22,1
52,51,1
53,40,1
53,20,1
53,24,1
53,26,1
54,40,1
54,41,1
54,12,1
54,47,1
54,23,1
54,27,1
55,10,1
55,12,1
55,15,1
55,19,1
56,11,1
56,19,1
56,43,1
57,13,1
57,56,1
57,20,1
57,48,1
57,49,1
58,14,1
58,7,1
58,30,1
58,49,1
58,50,1
58,51,1
58,52,1
60,59,1
61,59,1
62,59,1
62,60,1
63,59,1
63,60,1
63,62,1
64,58,1
64,51,1
64,52,1
65,64,1
65,58,1
65,51,1
66,64,1
66,28,1
66,30,1
67,64,1
67,65,1
67,66,1
67,30,1
68,64,1
68,58,1
68,65,1
69,64,1
69,58,1
69,65,1
69,51,1
70,64,1
70,66,1
70,30,1
71,7,1
71,68,1
71,70,1
72,71,1
72,28,1
72,66,1
72,49,1
72,70,1
73,71,1
73,72,1
73,66,1
73,30,1
73,31,1
74,71,1
74,72,1
74,73,1
74,66,1
74,30,1
74,31,1
75,71,1
75,72,1
75,73,1
75,74,1
75,30,1
75,31,1
75,70,1
76,71,1
76,72,1
76,75,1
76,66,1
76,30,1
76,31,1
76,49,1
76,53,1
77,71,1
77,75,1
77,58,1
77,30,1
77,19,1
77,31,1
77,76,1
78,71,1
78,72,1
78,74,1
78,75,1
78,31,1
79,71,1
79,72,1
79,74,1
79,75,1
79,30,1
80,71,1
80,72,1
80,66,1
80,30,1
81,71,1
82,71,1
82,72,1
82,73,1
82,74,1
82,75,1
82,30,1
82,31,1
82,76,1
83,71,1
83,73,1
83,75,1
83,30,1
83,76,1
84,72,1
84,74,1
84,75,1
84,66,1
84,73,1
84,60,1
84,30,1
84,62,1
84,76,1
84,79,1
84,81,1
84,82,1
84,83,1
85,72,1
85,7,1
85,58,1
85,65,1
85,66,1
86,72,1
86,73,1
86,66,1
86,84,1
86,60,1
86,30,1
86,61,1
86,76,1
86,81,1
87,72,1
87,74,1
87,84,1
87,83,1
88,72,1
88,74,1
88,66,1
88,84,1
89,72,1
89,73,1
89,66,1
89,84,1
89,86,1
89,88,1
90,72,1
90,66,1
90,70,1
91,72,1
91,74,1
91,75,1
91,31,1
91,90,1
91,79,1
92,72,1
92,73,1
92,75,1
93,73,1
93,66,1
93,86,1
93,30,1
94,73,1
94,84,1
94,93,1
95,73,1
95,94,1
95,61,1
96,73,1
96,66,1
96,84,1
96,94,1
97,73,1
97,66,1
97,84,1
97,86,1
97,96,1
97,81,1
98,73,1
98,74,1
98,87,1
98,91,1
99,73,1
99,74,1
99,66,1
99,84,1
99,93,1
99,60,1
99,59,1
99,30,1
99,62,1
99,63,1
99,90,1
100,73,1
100,66,1
100,84,1
100,86,1
100,96,1
100,98,1
100,99,1
100,62,1
100,79,1
100,91,1
100,83,1
101,84,1
101,86,1
101,94,1
101,100,1
101,61,1
102,93,1
102,94,1
102,95,1
102,46,1
103,67,1
104,67,1
104,69,1
104,103,1"""

edges_pb = set()
for line in polbooks_raw_lines.strip().split("\n"):
    parts = line.split(",")
    if len(parts) == 3:
        u, v = int(parts[0]), int(parts[1])
        edges_pb.add((min(u, v), max(u, v)))

with open(f"{base}/polbooks.edges", "w") as f:
    f.write("# Political Books Co-purchase Network\n")
    f.write("# 105 nodes (IDs 0-104), 441 edges, unit weight (all = 1.0)\n")
    f.write("# Domain: Recommendation / Political Science Network\n")
    f.write("# Source: V. Krebs (2004), compiled from Amazon.com co-purchase data\n")
    f.write("# Available: SuiteSparse Newman/polbooks; https://networks.skewed.de/net/polbooks\n")
    f.write("# Node labels in polbooks_nodes.txt\n")
    f.write("# Format: node_i node_j weight\n")
    for u, v in sorted(edges_pb):
        f.write(f"{u} {v} 1.0\n")

pb_nodes = set()
for u, v in edges_pb:
    pb_nodes.add(u); pb_nodes.add(v)
print(f"Polbooks: {len(edges_pb)} edges, {len(pb_nodes)} nodes")


# Write polbooks node label file
polbooks_node_labels = {
    0:"1000 Years for Revenge", 1:"Bush vs. the Beltway", 2:"Charlie Wilson's War",
    3:"Losing Bin Laden", 4:"Sleeping With the Devil", 5:"The Man Who Warned America",
    6:"Why America Slept", 7:"Ghost Wars", 8:"A National Party No More",
    9:"Bush Country", 10:"Dereliction of Duty", 11:"Legacy",
    12:"Off with Their Heads", 13:"Persecution", 14:"Rumsfeld's War",
    15:"Breakdown", 16:"Betrayal", 17:"Shut Up and Sing", 18:"Meant To Be",
    19:"The Right Man", 20:"Ten Minutes from Normal", 21:"Hillary's Scheme",
    22:"The French Betrayal of America", 23:"Tales from the Left Coast",
    24:"Hating America", 25:"The Third Terrorist", 26:"Endgame",
    27:"Spin Sisters", 28:"All the Shah's Men", 29:"Dangerous Dimplomacy",
    30:"The Price of Loyalty", 31:"House of Bush, House of Saud",
    32:"The Death of Right and Wrong", 33:"Useful Idiots",
    34:"The O'Reilly Factor", 35:"Let Freedom Ring", 36:"Those Who Trespass",
    37:"Bias", 38:"Slander", 39:"The Savage Nation",
    40:"Deliver Us from Evil", 41:"Give Me a Break", 42:"The Enemy Within",
    43:"The Real America", 44:"Who's Looking Out for You?",
    45:"The Official Handbook Vast Right Wing Conspiracy",
    46:"Power Plays", 47:"Arrogance", 48:"The Perfect Wife",
    49:"The Bushes", 50:"Things Worth Fighting For",
    51:"Surprise, Security, the American Experience", 52:"Allies",
    53:"Why Courage Matters", 54:"Hollywood Interrupted",
    55:"Fighting Back", 56:"We Will Prevail",
    57:"The Faith of George W Bush", 58:"Rise of the Vulcans",
    59:"Downsize This!", 60:"Stupid White Men",
    61:"Rush Limbaugh Is a Big Fat Idiot",
    62:"The Best Democracy Money Can Buy",
    63:"The Culture of Fear", 64:"America Unbound",
    65:"The Choice", 66:"The Great Unraveling", 67:"Rogue Nation",
    68:"Soft Power", 69:"Colossus", 70:"The Sorrows of Empire",
    71:"Against All Enemies", 72:"American Dynasty",
    73:"Big Lies", 74:"The Lies of George W. Bush",
    75:"Worse Than Watergate", 76:"Plan of Attack",
    77:"Bush at War", 78:"The New Pearl Harbor",
    79:"Bushwomen", 80:"The Bubble of American Supremacy",
    81:"Living History", 82:"The Politics of Truth",
    83:"Fanatics and Fools", 84:"Bushwhacked",
    85:"Disarming Iraq",
    86:"Lies and the Lying Liars Who Tell Them",
    87:"MoveOn's 50 Ways to Love Your Country",
    88:"The Buying of the President 2004",
    89:"Perfectly Legal", 90:"Hegemony or Survival",
    91:"The Exception to the Rulers", 92:"Freethinkers",
    93:"Had Enough?", 94:"It's Still the Economy, Stupid!",
    95:"We're Right They're Wrong", 96:"What Liberal Media?",
    97:"The Clinton Wars", 98:"Weapons of Mass Deception",
    99:"Dude, Where's My Country?", 100:"Thieves in High Places",
    101:"Shrub", 102:"Buck Up Suck Up",
    103:"The Future of Freedom", 104:"Empire",
}
with open(f"{base}/polbooks_nodes.txt", "w") as f:
    f.write("# Political Books node index -> book title\n")
    for idx, title in sorted(polbooks_node_labels.items()):
        f.write(f"{idx} {title}\n")

print("All 4 datasets written successfully.")
print(f"Output directory: {base}")
