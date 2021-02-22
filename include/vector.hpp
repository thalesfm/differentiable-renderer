#pragma once

#include <cstddef>
#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <type_traits>

namespace details {

template <typename ForwardIt1, typename ForwardIt2>
void add_ip(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2)
{
    std::transform(first1, last1, first2, first1,
        std::plus<typename std::iterator_traits<ForwardIt1>::value_type>());
}

}

template <typename T, std::size_t N>
class IVector {
private:
    std::array<T, N> m_data;

public:
    using value_type = T;
    using iterator = typename decltype(m_data)::iterator;
    using const_iterator = typename decltype(m_data)::const_iterator;

    IVector()
    { }

    IVector(std::initializer_list<T> init_list)
    {
        if (init_list.size() != N)
            throw std::runtime_error(
                "incorrect number of initializers for `Vector`");
        std::copy_n(init_list.begin(), N, data());
    }

    T& operator[](size_t pos)
    { return m_data[pos]; }

    const T& operator[](size_t pos) const
    { return m_data[pos]; }

    T *data()
    { return m_data.data(); }

    const T *data() const
    { return m_data.data(); }

    iterator begin()
    { return m_data.begin(); }

    const_iterator begin() const
    { return m_data.begin(); }

    iterator end()
    { return m_data.end(); }

    const_iterator end() const
    { return m_data.end(); }

    constexpr std::size_t size() const
    { return N; }

    virtual bool requires_grad() const = 0;

    virtual void backward(const IVector<T, N>& out_grad) = 0;
};

template <typename T, std::size_t N, bool RequiresGrad = false>
class Vector;

template <typename T, std::size_t N>
class Vector<T, N, false> : public IVector<T, N> {
public:
    Vector()
    { }

    Vector(std::initializer_list<T> init_list)
      : IVector<T, N>(init_list)
    { }

    bool requires_grad() const override
    { return false; }

    void backward(const IVector<T, N>& out_grad) override
    { }
};

template <typename T, std::size_t N>
class Vector<T, N, true> : public Vector<T, N> {
public:
    Vector()
    { }

    Vector(std::initializer_list<T> init_list)
      : IVector<T, N>(init_list)
    { }

    constexpr bool requires_grad() const override
    { return true; }

    void backward(const IVector<T, N>& out_grad) override
    { details::add_ip(m_grad.begin(), m_grad.end(), out_grad.begin()); }

private:
    std::array<T, N> m_grad;
};

template <typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const IVector<T, N>& v)
{
    os << "[";
    for (size_t i = 0; i < v.size()-1; ++i)
        os << v[i] << ", ";
    if (v.size() > 0)
        os << v[v.size()-1];
    return os << "]";
}

namespace details {

std::false_type is_ivector_impl(...);

template <typename T, std::size_t N>
std::true_type is_vector_impl(IVector<T, N>*);

template <typename T>
using is_vector = decltype(is_vector_impl(std::declval<T*>()));

template <typename T, std::size_t N>
IVector<T, N> base_vector_impl(IVector<T, N>*);

template <typename T>
struct base_vector {
    using type = decltype(base_vector_impl(std::declval<T*>()));
};

template <typename T>
struct vector_size_impl;

template <typename T, std::size_t N>
struct vector_size_impl<IVector<T, N>> :
    std::integral_constant<std::size_t, N> { };

template <typename T>
struct vector_size : vector_size_impl<typename base_vector<T>::type> { };

template <typename LHS, typename RHS>
struct valid_binop_args :
    std::integral_constant<bool,
        is_vector<LHS>::value && is_vector<RHS>::value &&
        std::is_same<typename LHS::value_type, typename RHS::value_type>::value &&
        vector_size<LHS>::value == vector_size<RHS>::value> { };

}

template <typename T, std::size_t N>
class AddResult : public IVector<T, N> {
public:
    AddResult(const IVector<T, N>& lhs, const IVector<T, N>& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
        // details::add_ip(begin(), end(), 
    }

    bool requires_grad() override
    { return true; }

    void backward(const IVector<T, N>& out_grad)
    { }
private:
    const IVector<T, N>& m_lhs;
    const IVector<T, N>& m_rhs;
};

template <typename T, std::size_t N>
Vector<T, N> operator+(
    const Vector<T, N, false>& lhs, const Vector<T, N, false>& rhs)
{
    Vector<T, N, false> r = lhs;
    details::add_ip(r.begin(), r.end(), rhs.begin());
    return r;
}

template <typename LHS, typename RHS,
    typename = std::enable_if<details::valid_binop_args<LHS, RHS>::value>>
std::unique_ptr<typename details::base_vector<LHS>::type> operator+(
    const LHS& lhs, const RHS& rhs)
{
    AddResult<typename LHS::value_type, details::vector_size<LHS>::value> r;
    return std::make_unique(r);
}

template <typename T, std::size_t N>
std::unique_ptr<IVector<T, N>> operator+(
    const IVector<T, N>& lhs, const IVector<T, N>& rhs)
{
    return std::make_unique<AddResult<T, N>>(lhs, rhs);
}

/*
void is_vector_test()
{
    class ConvertibleToVector {
    public:
        operator Vector<char, 1>() const
        { return Vector<char, 1>(); }
    };

    Vector<char, 1> v = ConvertibleToVector();
    std::cout << "is_vector<double>::value = " << is_vector<double>::value << std::endl;
    std::cout << "is_vector<Vector<float, 3>>::value = " << is_vector<Vector<float, 3>>::value << std::endl;
    std::cout << "is_vector<Variable<float, 3>>::value = " << is_vector<Variable<float, 3>>::value << std::endl;
    std::cout << "is_vector<ConvertibleToVector>::value = " << is_vector<ConvertibleToVector>::value << std::endl;
}
*/

void test()
{
    Vector<float, 3> v1, v2;
    Vector<float, 3, true> var;
    // auto r1 = v1 + v2;
    // auto r2 = v1 + var;
}
