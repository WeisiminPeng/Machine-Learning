

from PIL import Image
import pydotplus as pdp
from GBDT.decision_tree import Node, Tree
import os
import matplotlib.pyplot as plt
##import pdb


def plot_tree_model(tree: Tree, max_depth: int, iter: int):
    """
           desplay single decision tree
    :param tree: Generated decision tree
    :param max_depth: max depth of decision tree
    :param iter: the number of decision tree
    :return:
    """
    
    root = tree.root_node
    res = []
    # Obtain the parent-child node relationship of the decision tree by traversing, 
    # optional traversal level traversal and traversal_preorder traversal
    traversal(root, res)

    # Got all node point
    nodes = {}   #build a array
    index = 0
    for i in res:
        p, c = i[0], i[1]
        if p not in nodes.values():    
            nodes[index] = p
            index = index + 1
        if c not in nodes.values():
            nodes[index] = c
            index = index + 1     ##index：Number of nodes

    # Display the decision tree through dot syntax
    edges = ''
    node = ''
    # pdb.set_trace()
    # Show node hierarchy
    for depth in range(max_depth):
        for nodepair in res:
            if nodepair[0].deep == depth:
                # p, c are the parent node and child node in the node pair respectively
                p_, c_ = nodepair[0], nodepair[1]
                l = len([i for i in range(len(c.data_index)) if c.data_index[i] is True])
                pname = str(list(nodes.keys())[list(nodes.values()).index(p_)])
                cname = str(list(nodes.keys())[list(nodes.values()).index(c_)])
                if l > 0:
                    edges = edges + pname + '->' + cname + '[label=\"' + str(p_.split_feature) + (
                        '<' if p_.left_child == c_ else '>=') + str(p_.split_value) + '\"]' + ';\n'

                node = node + pname + '[width=1,height=0.5,color=lemonchiffon,style=filled,shape=ellipse,label=\"Data\"];\n' + \
                       (cname + '[width=1,height=0.5,color=lemonchiffon,style=filled,shape=ellipse,label=\"Data\"];\n')
                if c_.is_leaf and l > 0:
                    edges = edges + cname + '->' + cname + 'p[style=dotted];\n'
                    node = node + cname + 'p[width=1,height=0.5,color=lightskyblue,style=filled,shape=box,label=\"' + str(
                        "{:.4f}".format(c_.predict_value)) + '\"];\n'
            else:
                continue
        dot = '''digraph g {\n''' + edges + node + '''}'''
        graph = pdp.graph_from_dot_data(dot)
        # Save picture and pyplot display
        graph.write_png('results/NO.{}_tree.jpg'.format(iter))
        img = Image.open('results/NO.{}_tree.jpg'.format(iter))
        img = img.resize((1024, 700), Image.ANTIALIAS)
        plt.ion()
        plt.figure(1, figsize=(30, 20))
        plt.axis('off')
        plt.title('NO.{} tree'.format(iter))
        plt.rcParams['figure.figsize'] = (30.0, 20.0)
        plt.imshow(img)
        plt.pause(0.01)


def plot_all_trees(numberOfTrees: int):
    '''
    Collect all the generated decision trees into one picture for display
    :param numberOfTrees: the number of decision tree
    :return:
    '''
    # Each row displays 3 decision trees. 
    # The number of rows is determined by the number of decision trees.
    if numberOfTrees / 3 - int(numberOfTrees / 3) > 0.000001:    #树个数不是三的整数就加一排
        rows = int(numberOfTrees / 3)+1
    else:
        rows = int(numberOfTrees / 3)
    # Use subplot to display all decision trees in one figure
    plt.figure(1, figsize=(30,20))
    plt.axis('off')
    try:
        for index in range(1, numberOfTrees + 1):
            path = os.path.join('results', 'NO.{}_tree.jpg'.format(index))
            plt.subplot(rows, 3, index)
            img = Image.open(path)
            img = img.resize((1000, 800), Image.ANTIALIAS)
            plt.axis('off')
            plt.title('NO.{} tree'.format(index))
            plt.imshow(img)
        plt.savefig('results/all_trees.jpg', dpi=300)
        plt.show()
        # Since pyplot picture pixels are not very high, 
        # use the method to generate high-quality pictures
        image_compose(numberOfTrees)
    except Exception as e:
        raise e


def image_compose(numberOfTrees: int):
    '''
    Stitch the picture of numberOfTrees decision tree into one picture
    :param numberOfTrees: number of decision tree
    :return:
    '''

    png_to_compose = []
    # Get the size of each picture
    for index in range(1,numberOfTrees+1):
        png_to_compose.append('NO.{}_tree.jpg'.format(index))
    try:
        path = os.path.join('results', png_to_compose[0])
        shape = Image.open(path).size
    except Exception as e:
        raise e
    IMAGE_WIDTH = shape[0]
    IMAGE_HEIGET = shape[1]
    IMAGE_COLUMN = 3

    if len(png_to_compose)/IMAGE_COLUMN - int(len(png_to_compose)/IMAGE_COLUMN) > 0.0000001:
        IMAGE_ROW = int(len(png_to_compose)/IMAGE_COLUMN)+1
    else:
        IMAGE_ROW = int(len(png_to_compose) / IMAGE_COLUMN)
    # Create a new picture for stitching
    to_image = Image.new('RGB', (IMAGE_COLUMN*IMAGE_WIDTH, IMAGE_ROW*IMAGE_HEIGET), '#FFFFFF')
    # stitch picture
    for y in  range(IMAGE_ROW):
        for x in range(IMAGE_COLUMN):
            if y*IMAGE_COLUMN+x+1 > len(png_to_compose):
                break
            path = os.path.join('results', 'NO.'+str(y*IMAGE_COLUMN+x+1)+'_tree.jpg')
            from_image = Image.open(path)
            to_image.paste(from_image, (x*IMAGE_WIDTH, y*IMAGE_HEIGET))

    to_image.save('results/all_trees_high_quality.jpg')


def traversal_preorder(root: Node, res: list):  
    '''
    First traverse the decision tree to obtain the parent-child relationship between nodes
     :param root: the root node of the decision tree
     :param res: store a list of node pairs (parent node, child node)
     :return: res
    '''
    if root is None:
        return
    if root.left_child is not None:
        res.append([root, root.left_child])
        traversal_preorder(root.left_child, res)
    if root.right_child is not None:
        res.append([root, root.right_child])
        traversal_preorder(root.right_child, res)


def traversal(root: Node, res: list):   
    '''
    Hierarchically traverse the decision tree to obtain the parent-child relationship between nodes
     :param root: the root node of the decision tree
     :param res: store a list of node pairs (parent node, child node)
     :return: res
    '''
    outList = []
    queue = [root]
    while queue != [] and root:
        outList.append(queue[0].data_index)
        if queue[0].left_child != None:
            queue.append(queue[0].left_child)
            res.append([queue[0], queue[0].left_child])
        if queue[0].right_child != None:
            queue.append(queue[0].right_child)
            res.append([queue[0], queue[0].right_child])
        queue.pop(0)


if __name__ =="__main__":
    plot_all_trees(10)
    # image_compose(10)

