#include "BTrees.hpp"
#include <iostream>
#include <random>
#include <functional>
#include <map>
using namespace std;

// Forward-declare BTree objects
template <size_t N, typename T> ostream& operator<<(ostream& o, BTree<N, T> const& v);

template <size_t NodeSize> size_t testBTreeWithNodeSize(size_t elements, size_t runs) {
  // generate shuffled unique values
  auto input = new long[elements];
  for(size_t i = 0; i < elements; i++)
    input[i] = i;

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(input, input + elements, g);

  // create and fill BTree
  BTree<NodeSize> b(input[0]);

  for(size_t i = 1; i < elements; i++) {
    auto& it = input[i];
    b.insert(it);
  }

  // reshuffle input values
  std::shuffle(input, input + elements, g);

  // Lookup
  auto sum = 0ul;
  for(size_t run = 0; run < runs; run++)
    for(size_t i = 0; i < elements; i++)
      sum += b.count(input[i]);

  return sum;
}

int main(int argc, char* argv[]) {
  if(argc < 4) {
    cerr << "usage: BTrees ${nodeSize(4, 8, 16, 32, 64 or 128)} ${elementsInTree} ${runs}" << endl;
    exit(1);
  }
  auto nodeSize = atoi(argv[1]);
  auto elements = atoi(argv[2]);
  auto runs = atoi(argv[3]);

  cout << map<int, function<size_t(size_t, size_t)>>({
                                                         {4, testBTreeWithNodeSize<4>},
                                                         {8, testBTreeWithNodeSize<8>},
                                                         {16, testBTreeWithNodeSize<16>},
                                                         {32, testBTreeWithNodeSize<32>},
                                                         {64, testBTreeWithNodeSize<64>},
                                                         {128, testBTreeWithNodeSize<128>},
                                                     })
              .at(nodeSize)(elements, runs)
       << endl;
  return 0;
}
