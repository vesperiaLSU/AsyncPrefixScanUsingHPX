#ifndef Guard_PrefixScan_h
#define Guard_PrefixScan_h

#include <iterator>
#include <vector>

//Sequential//
template<typename InputIterator, typename OutputIterator, typename T = std::iterator_traits<InputIterator>::value_type, typename BinaryOp = std::plus<T>>
OutputIterator exclusive_scan(InputIterator first, InputIterator last, OutputIterator dest, T init = T(), BinaryOp op = BinaryOp()){
    for (/**/; first != last; (void) ++first, ++dest)
    {
        *dest = init;
        init = op(init, *first);
    }
    return dest;
}

template<typename InputIterator, typename OutputIterator, typename T = std::iterator_traits<InputIterator>::value_type, typename BinaryOp = std::plus<T>>
OutputIterator inclusive_scan(InputIterator first, InputIterator last, OutputIterator dest, T init = T(), BinaryOp op = BinaryOp()){
    for (/**/; first != last; (void) ++first, ++dest)
    {
        init = op(init, *first);
        *dest = init;
    }
    return dest;
}

template<typename OutputIterator>
OutputIterator input_generation(size_t count, OutputIterator dest){
    for (size_t i = 0; i < count; i++)
    {
        double x = rand() % 100;
        //cout << x << " ";
        *dest++ = x;
    }
    cout << "End of generation" << endl;
    return dest;
}

#endif
