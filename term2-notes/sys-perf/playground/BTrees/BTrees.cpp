#include <cstdlib>
#include <utility>
#include <iostream>
#include <optional>
#include "BTrees.hpp"
using namespace std;

/**
 * A Node in the Tree,
 *
 * NodeSize is the out-degree (the number of children),
 *
 * T is the type of the held value
 */
template <size_t NodeSize, typename T> struct BTreeNode {
  /**
   * An element in the node consisting of a pivot and the subtree to its left
   */
  struct NodeElement {
    /**
     * The node to the left of the pivot
     */
    BTreeNode<NodeSize, T>* childToLeft{};
    /**
     * The pivot, we're using the optional template to encode pivots slots that don't contain a
     * value
     */
    optional<T> pivot{};
  };

  /**
   * A struct to (temporarily) hold newly created splits
   */
  struct NewSplit {
    /**
     * the newly introduced pivot, this needs merging into a node
     */
    T newPivot;
    /**
     * the newly create subtree to the right of the pivot
     */
    BTreeNode<NodeSize, T>* childToTheRightOfPivot;
  };

  /**
   * since the out-degree is NodeSize, we have NodeSize-1 pivots
   */
  NodeElement pivots[NodeSize - 1]{};
  /**
   * the child to the very right of the node, it is not associated/paired with a pivot
   */
  BTreeNode<NodeSize, T>* rightMost = nullptr;

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  ////////// There is no need for you to change anything above this line//////////
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  NewSplit splitNode(T first, BTreeNode<NodeSize, T>* newChild) {
    auto newNode = new BTreeNode<NodeSize, T>;
    for(size_t i = 0; i < (NodeSize - 1) / 2; i++) {
      newNode->pivots[i] = pivots[i + NodeSize / 2];
      pivots[i + NodeSize / 2] = {};
    }
    newNode->pivots[(NodeSize - 1) / 2].childToLeft = rightMost;
    rightMost = {};
    auto splitElement = pivots[NodeSize / 2 - 1];
    pivots[NodeSize / 2 - 1].pivot = {};
    (splitElement.pivot < first ? newNode : this)->mergeValueIntoNode(first, newChild);
    return {*splitElement.pivot, newNode};
  }

  NewSplit mergeValueIntoNode(T first, BTreeNode<NodeSize, T>* second, bool isLeaf = false) {
    if(!rightMost) { // node not full yet
      rightMost = pivots[NodeSize - 2].childToLeft;
      int slot = NodeSize - 3;
      for(; slot >= 0 && (!pivots[slot].pivot || pivots[slot].pivot > first); slot--)
        pivots[slot + 1] = pivots[slot];
      pivots[slot + 1].pivot = first;
      pivots[slot + 2].childToLeft = second;
      return {{}, nullptr};
    } else { // node full
      return splitNode(first, second);
    }
  }

  /**
   * Inserts the value into the Node (or any of it children)
   *
   *
   * If the node was split, returns a new pivot and a new right child
   */
  NewSplit insert(T v, size_t level = 0) {
    if(pivots[0].childToLeft) { // inner node
      NewSplit newChild{};
      for(size_t i = 0; i < NodeSize - 1; i++) {
        if(!pivots[i].pivot || pivots[i].pivot > v) { // found pivot
          newChild = pivots[i].childToLeft->insert(v, level + 1);
          break;
        }
        if(i == NodeSize - 2)
          newChild = rightMost->insert(v, level + 1);
      }
      if(!newChild.childToTheRightOfPivot) // no split occured
        return {{}, nullptr};
      else // split occured
        return mergeValueIntoNode(newChild.newPivot, newChild.childToTheRightOfPivot);
    } else { // leaf node
      if(pivots[NodeSize - 2].pivot) {
        return splitNode(v, nullptr);
      } else { // there is space in the node
        size_t toLeft = 0;
        while(pivots[toLeft].pivot && pivots[toLeft].pivot < v)
          toLeft++;
        rightMost = pivots[NodeSize - 2].childToLeft;
        for(size_t toRight = NodeSize - 2; toRight > toLeft; toRight--) {
          pivots[toRight] = pivots[toRight - 1];
        }
        pivots[toLeft].pivot = v;
        return {{}, nullptr};
      }
    }
  }

  /**
   * Count the number of occurences of the value v
   *
   * You can assume uniqueness of the values v, thus this function
   * returns either 0 of 1
   */
  size_t count(T v) {
    size_t i = 0;
    for(; pivots[i].pivot && i < NodeSize - 1; i++) {
      if(pivots[i].pivot == v)
        return 1;
      else if(pivots[i].pivot > v) {
        if(pivots[i].childToLeft == nullptr)
          return 0;
        else
          return pivots[i].childToLeft->count(v);
      }
    }
    if(i < NodeSize - 1)
      return pivots[i].childToLeft ? pivots[i].childToLeft->count(v) : 0;
    else
      return rightMost ? rightMost->count(v) : 0;
  }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////// There is no need for you to change anything below this line//////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <size_t NodeSize, typename T> void BTree<NodeSize, T>::insert(T v) {
  auto newNode = root->insert(v);
  auto oldRoot = root;
  if(newNode.childToTheRightOfPivot) {
    root = new BTreeNode<NodeSize, T>{};
    root->pivots[0].pivot = newNode.newPivot;
    root->pivots[0].childToLeft = oldRoot;
    root->pivots[1].childToLeft = newNode.childToTheRightOfPivot;
  }
}

template <size_t NodeSize, typename T>
BTree<NodeSize, T>::BTree(T v)
    : root(new BTreeNode<NodeSize, T>) {
  root->pivots[0].pivot = v;
}

template <size_t NodeSize, typename T> size_t BTree<NodeSize, T>::count(T v) {
  return root->count(v);
}

template <size_t NodeSize, typename T>
ostream& operator<<(ostream& o, typename BTreeNode<NodeSize, T>::NodeElement const& v) {
  return v.childToLeft ? (o << *(v.childToLeft) << ", " << *v.pivot) : (o << *v.pivot);
}

template <size_t NodeSize, typename T>
ostream& operator<<(ostream& o, BTreeNode<NodeSize, T> const& v) {
  o << "[";
  operator<<<NodeSize, T>(o, v.pivots[0]);
  for(auto it = next(begin(v.pivots)); it != end(v.pivots); ++it) {
    if(it->pivot)
      operator<<<NodeSize, T>(o << ", ", *it);
    else if(it->childToLeft)
      o << ", " << *it->childToLeft;
  }
  return (v.rightMost ? (o << ", " << *v.rightMost) : o) << "]";
}

template <size_t NodeSize, typename T>
ostream& operator<<(ostream& o, BTree<NodeSize, T> const& v) {
  return o << *v.root << endl;
}

template class BTree<4>;
template ostream& operator<<(ostream& o, BTree<4, long> const& v);
template class BTree<8>;
template ostream& operator<<(ostream& o, BTree<8, long> const& v);
template class BTree<16>;
template ostream& operator<<(ostream& o, BTree<16, long> const& v);
template class BTree<32>;
template ostream& operator<<(ostream& o, BTree<32, long> const& v);
template class BTree<64>;
template ostream& operator<<(ostream& o, BTree<64, long> const& v);
template class BTree<128>;
template ostream& operator<<(ostream& o, BTree<128, long> const& v);
