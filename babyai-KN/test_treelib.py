from treelib import Node, Tree
tree = Tree()

K, N = 3, 3

index = 0 
tree.create_node(str(index), str(index)) # root node

for i in range(1, N, 1):#### parent ：：：(index-1)//K # the index for cur layer
    next_i = i+1
    layer_begin = (K**(next_i-1)-1)//(K-1)
    layer_end = layer_begin + K**i   # [layer_begin, layer_end)
    for j in range(layer_begin, layer_end, 1):
        tree.create_node(str(j), str(j), parent=str((j-1)//K))

tree.show()
print(tree.paths_to_leaves())

import pdb
pdb.set_trace()

print()

paths = [list(map(int, i)) for i in tree.paths_to_leaves()]
