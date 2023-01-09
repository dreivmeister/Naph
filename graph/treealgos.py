from node import BinaryTreeNode

def preorder(root, nodes):
    if root is None:
        return
    nodes.append(root.val)
    preorder(root.left_child,nodes)
    preorder(root.right_child,nodes)
    
def inorder(root, nodes):
    if root is None:
        return
    inorder(root.left_child,nodes)
    nodes.append(root.val)
    inorder(root.right_child,nodes)
    
def postorder(root, nodes):
    if root is None:
        return
    postorder(root.left_child,nodes)
    postorder(root.right_child,nodes)
    nodes.append(root.val)

def eulerian_tour(root, tour):
    if root is None:
        return
    tour.append(root.val)
    
    eulerian_tour(root.left_child, tour)
    tour.append(root.val)
    
    eulerian_tour(root.right_child, tour)
    tour.append(root.val)
    
def max_path_sum(root):
    max_path = -float('inf')
    
    def trav(node):
        nonlocal max_path
        
        if not node:
            return 0
        
        left = max(trav(node.left_child),0)
        right = max(trav(node.right_child),0)
        
        max_path = max(max_path,left+right+node.val)
        
        return max(left+node.val,right+node.val)
    
    trav(root)
    return max_path
    

if __name__=="__main__":
    # nc = BinaryTreeNode(6)
    # ne = BinaryTreeNode(7)
    # nh = BinaryTreeNode(8)
    # na = BinaryTreeNode(3)
    # nd = BinaryTreeNode(4,nc,ne)
    # nb = BinaryTreeNode(1,na,nd)
    # ni = BinaryTreeNode(5,nh)
    # ng = BinaryTreeNode(2,None,ni)
    # nf = BinaryTreeNode(0,nb,ng)
    
    
    root = BinaryTreeNode(10)
    root.left_child = BinaryTreeNode(2)
    root.right_child = BinaryTreeNode(10)
    root.left_child.left_child = BinaryTreeNode(20)
    root.left_child.right_child = BinaryTreeNode(1)
    root.right_child.right_child = BinaryTreeNode(-25)
    root.right_child.right_child.left_child = BinaryTreeNode(3)
    root.right_child.right_child.right_child = BinaryTreeNode(4)
    
    
    #nodelist = []
    #postorder(nf,nodelist)
    #print(nodelist) #F,B,A,D,C,E,G,I,H
    
    # tour = []
    # eulerian_tour(nf,tour)
    # print(tour) #0,1,3,1,4,6,4,7,4,1,0,2,5,8,5,2,0
    
    print(max_path_sum(root))
    
    