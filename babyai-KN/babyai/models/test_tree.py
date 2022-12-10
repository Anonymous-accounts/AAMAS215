# coding=utf-8
from treelib import Tree, Node
 
tree = Tree()
tree.show()

K, N = 3, 2
ids = ['root']
parents = []

for i in range(N):
    for j in range(K):
        ids.append(str(i+1) + '-layer-' + str(j))
        parents.append(ids[i])
    


tree.create_node(tag='1', identifier='root', data=5)
tree.create_node(tag='1', identifier='1layer-0', parent='root', data=10)
tree.create_node(tag='3', identifier='1layer-1', parent='root', data=10)
tree.create_node(tag='2', identifier='1layer-2', parent='root', data=10)

tree.create_node(identifier='2layer-0-0', parent='1layer-0', data=10)
tree.create_node(identifier='2layer-0-1', parent='1layer-0', data=10)
tree.create_node(identifier='2layer-0-2', parent='1layer-0', data=10)

tree.create_node(identifier='2layer-1-0', parent='1layer-1', data=10)
tree.create_node(identifier='2layer-1-1', parent='1layer-1', data=10)
tree.create_node(identifier='2layer-1-2', parent='1layer-1', data=10)


tree.create_node(identifier='2layer-2-0', parent='1layer-2', data=10)
tree.create_node(identifier='2layer-2-1', parent='1layer-2', data=10)
tree.create_node(identifier='2layer-2-2', parent='1layer-2', data=10)

tree.show()


print(tree.paths_to_leaves())
print(tree.get_node('1layer-0').data)
print(tree.get_node('1layer-0').tag)





