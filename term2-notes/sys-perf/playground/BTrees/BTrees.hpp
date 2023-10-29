#include <cstdlib>
template <size_t NodeSize, typename T = long> class BTreeNode;
template <size_t NodeSize, typename T = long> class BTree {
public:
	BTreeNode<NodeSize, T>* root;
	void insert(T v);
	size_t count(T v);
	BTree(T v);
};
