#pragma once

#include <cstddef>
#include <algorithm>
#include <array>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <typeinfo>
#include <type_traits>
#include "complex.hpp"

namespace drt {

template <typename T, std::size_t N, bool Autograd = false>
class Vector;

using Vec1 = Vector<double, 1>;
using Vec2 = Vector<double, 2>;
using Vec3 = Vector<double, 3>;
using Var1 = Vector<double, 1, true>;
using Var2 = Vector<double, 2, true>;
using Var3 = Vector<double, 3, true>;
using Vec1f = Vector<float, 1>;
using Vec2f = Vector<float, 2>;
using Vec3f = Vector<float, 3>;
using Var1f = Vector<float, 1, true>;
using Var2f = Vector<float, 2, true>;
using Var3f = Vector<float, 3, true>;

template <typename T, std::size_t N>
class Vector<T, N> {
public:
    using iterator = typename std::array<T, N>::iterator;
    using const_iterator = typename std::array<T, N>::const_iterator;

    Vector() = default;

    explicit Vector(T value)
    { m_data.fill(value); }

    Vector(std::initializer_list<T> init)
    {
        if (init.size() != N)
            throw std::runtime_error(
                "incorrect number of initializers for `Vector`");
        std::copy(init.begin(), init.end(), begin());
    }

    T& operator[](std::size_t pos)
    { return m_data[pos]; }

    const T& operator[](std::size_t pos) const
    { return m_data[pos]; }

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

    Vector& operator+=(const Vector& rhs)
    {
        std::transform(begin(), end(), rhs.begin(), begin(),
            [](const T& x, const T& y) { return x + y; });
        return *this;
    }

    Vector& operator-=(const Vector& rhs)
    {
        std::transform(begin(), end(), rhs.begin(), begin(),
            [](T x, T y) { return x - y; });
        return *this;
    }

    Vector& operator*=(const Vector& rhs)
    {
        std::transform(begin(), end(), rhs.begin(), begin(),
            [](T x, T y) { return x * y; });
        return *this;
    }

    Vector& operator*=(T s)
    {
        std::transform(begin(), end(), begin(),
            [=](T x) { return x * s; });
        return *this;
    }

    Vector& operator/=(const Vector& rhs)
    {
        std::transform(begin(), end(), rhs.begin(), begin(),
            [](T x, T y) { return x / y; });
        return *this;
    }

    Vector& operator/=(T s)
    {
        std::transform(begin(), end(), begin(),
            [=](T x) { return x / s; });
        return *this;
    }

private:
    std::array<T, N> m_data;
};

namespace internal {

template <typename T, std::size_t N>
class AutogradNode : public Vector<T, N> {
public:
    using Vector<T, N>::Vector;

    AutogradNode(const Vector<T, N>& v) : Vector<T, N>(v) { }

    virtual ~AutogradNode() { }

    virtual Vector<T, N>& grad()
    { throw std::runtime_error("Vector has no gradient (not a variable)"); }

    virtual const Vector<T, N>& grad() const
    { throw std::runtime_error("Vector has no gradient (not a variable)"); }

    virtual bool requires_grad() const = 0;

    virtual void backward(const Vector<T, N>& grad) const = 0;
};

template <typename T, std::size_t N>
class ConstantNode : public AutogradNode<T, N> {
public:
    ConstantNode(const Vector<T, N>& v) : AutogradNode<T, N>(v) { }

    bool requires_grad() const override
    { return false; }

    void backward(const Vector<T, N>& grad) const override { }
};

template <typename T, std::size_t N>
class VariableNode : public AutogradNode<T, N> {
public:
    using AutogradNode<T, N>::AutogradNode;

    Vector<T, N>& grad() override
    { return m_grad; }

    const Vector<T, N>& grad() const override
    { return m_grad; }

    bool requires_grad() const override
    { return true; }

    void backward(const Vector<T, N>& grad) const override
    { m_grad += grad; }

private:
    mutable Vector<T, N> m_grad;
};

template <typename T, std::size_t N, typename Backward>
class BackwardNode : public AutogradNode<T, N> {
public:
    BackwardNode(const Vector<T, N>& v, const Backward& backward)
      : AutogradNode<T, N>(v), m_backward(backward) { }

    bool requires_grad() const override
    { return true; }

    void backward(const Vector<T, N>& grad) const override
    { m_backward(grad); }

private:
    typename std::decay<Backward>::type m_backward;
};

} // namespace internal

template <typename T, std::size_t N>
class Vector<T, N, true> {
public:
    // Default constructor assumes requires_grad = true
    // due to ambiguities related to other constructors
    Vector() : Vector(Vector<T, N>(), true) { }

    explicit Vector(T value, bool requires_grad = false)
      : Vector(Vector<T, N>(value), requires_grad) { }

    Vector(std::initializer_list<T> init, bool requires_grad = false)
      : Vector(Vector<T, N>(init), requires_grad) { }

    Vector(const Vector<T, N>& v, bool requires_grad = false)
    {
        if (requires_grad)
            m_ptr = std::make_shared<internal::VariableNode<T, N>>(v);
        else
            m_ptr = std::make_shared<internal::ConstantNode<T, N>>(v);
    }

    template <typename Backward>
    Vector(const Vector<T, N>& v, const Backward& backward)
      : m_ptr(new internal::BackwardNode<T, N, Backward>(v, backward)) { }

    // WARN: Potentially unsafe!
    T& operator[](std::size_t pos)
    { return (*m_ptr)[pos]; }

    const T& operator[](std::size_t pos) const
    { return (*m_ptr)[pos]; }

    constexpr std::size_t size() const
    { return N; }

    // WARN: Potentially unsafe!
    Vector<T, N>& detach()
    { return *m_ptr; }

    const Vector<T, N>& detach() const
    { return *m_ptr; }

    Vector<T, N>& grad()
    { return m_ptr->grad(); }

    const Vector<T, N>& grad() const
    { return m_ptr->grad(); }

    bool requires_grad() const
    { return m_ptr->requires_grad(); }

    void backward(const Vector<T, N>& grad) const
    { m_ptr->backward(grad); }

    Vector<T, N, true>& operator+=(const Vector<T, N, true>& rhs)
    { return *this = *this + rhs; }

    Vector<T, N, true>& operator-=(const Vector<T, N, true>& rhs)
    { return *this = *this - rhs; }

    Vector<T, N, true>& operator*=(const Vector<T, N, true>& rhs)
    { return *this = *this * rhs; }

    Vector<T, N, true>& operator*=(T s)
    { return *this = *this * s; }

    Vector<T, N, true>& operator/=(const Vector<T, N, true>& rhs)
    { return *this = *this / rhs; }

    Vector<T, N, true>& operator/=(T s)
    { return *this = *this / s; }

private:
    std::shared_ptr<internal::AutogradNode<T, N>> m_ptr;
};

namespace internal {

template <typename T, std::size_t N>
inline Vector<T, N>& detach(Vector<T, N>& v)
{ return v; }

template <typename T, std::size_t N>
inline const Vector<T, N>& detach(const Vector<T, N>& v)
{ return v; }

template <typename T, std::size_t N>
inline Vector<T, N>& detach(Vector<T, N, true>& v)
{ return v.detach(); }

template <typename T, std::size_t N>
inline const Vector<T, N>& detach(const Vector<T, N, true>& v)
{ return v.detach(); }

template <typename T, std::size_t N>
inline constexpr bool requires_grad(const Vector<T, N>& v)
{ return false; }

template <typename T, std::size_t N>
inline bool requires_grad(const Vector<T, N, true>& v)
{ return v.requires_grad(); }

template <typename T, std::size_t N>
inline void backward(Vector<T, N>& v, const Vector<T, N>& grad) { }

template <typename T, std::size_t N>
inline void backward(Vector<T, N, true>& v, const Vector<T, N>& grad)
{ v.backward(grad); }

template <typename T, std::size_t N>
struct AddBackward {
    void operator()(const Vector<T, N>& grad) const
    {
        lhs.backward(grad);
        rhs.backward(grad);
    }

    Vector<T, N, true> lhs, rhs;
};

template <typename T, std::size_t N>
struct SubBackward {
    void operator()(const Vector<T, N>& grad) const
    {
        lhs.backward(grad);
        rhs.backward(-grad);
    }

    Vector<T, N, true> lhs, rhs;
};

template <typename T, std::size_t N>
struct MulBackward {
    void operator()(const Vector<T, N>& grad) const
    {
        lhs.backward(rhs.detach() * grad);
        rhs.backward(lhs.detach() * grad);
    }

    Vector<T, N, true> lhs, rhs;
};

template <typename T, std::size_t N>
struct ScalarMulBackward {
    void operator()(const Vector<T, N>& grad) const
    { v.backward(s * grad); }

    T s;
    Vector<T, N, true> v;
};

template <typename T, std::size_t N>
struct DivBackward {
    void operator()(const Vector<T, N>& grad) const
    {
        lhs.backward(grad / rhs.detach());
        rhs.backward(-lhs.detach() * grad / (rhs.detach() * rhs.detach()));
    }

    Vector<T, N, true> lhs, rhs;
};

template <typename T, std::size_t N>
struct ScalarDivBackward {
    void operator()(const Vector<T, N>& grad) const
    { v.backward(grad / s); }

    Vector<T, N, true> v;
    T s;
};

} // namespace internal

template <typename T, std::size_t N, bool Ag>
inline Vector<T, N, Ag> operator-(const Vector<T, N, Ag>& v)
{ return -1 * v; }

template <typename T, std::size_t N>
inline Vector<T, N> operator+(const Vector<T, N>& lhs, const Vector<T, N>& rhs)
{ return Vector<T, N>(lhs) += rhs; }

template <typename T, std::size_t N, bool Ag1, bool Ag2,
          typename = typename std::enable_if<Ag1 || Ag2>::type>
inline Vector<T, N, true> operator+(const Vector<T, N, Ag1>& lhs,
                                    const Vector<T, N, Ag2>& rhs)
{
    auto r = internal::detach(lhs) + internal::detach(rhs);
    if (!internal::requires_grad(lhs) && !internal::requires_grad(rhs))
        return r;
    return Vector<T, N, true>(r, internal::AddBackward<T, N>{lhs, rhs});
}

template <typename T, std::size_t N>
inline Vector<T, N> operator-(const Vector<T, N>& lhs, const Vector<T, N>& rhs)
{ return Vector<T, N>(lhs) -= rhs; }

template <typename T, std::size_t N, bool Ag1, bool Ag2,
          typename = typename std::enable_if<Ag1 || Ag2>::type>
inline Vector<T, N, true> operator-(const Vector<T, N, Ag1>& lhs,
                                    const Vector<T, N, Ag2>& rhs)
{
    auto r = internal::detach(lhs) - internal::detach(rhs);
    if (!internal::requires_grad(lhs) && !internal::requires_grad(rhs))
        return r;
    return Vector<T, N, true>(r, internal::SubBackward<T, N>{lhs, rhs});
}

template <typename T, std::size_t N>
inline Vector<T, N> operator*(const Vector<T, N>& lhs, const Vector<T, N>& rhs)
{ return Vector<T, N>(lhs) *= rhs; }

template <typename T, std::size_t N, bool Ag1, bool Ag2,
          typename = typename std::enable_if<Ag1 || Ag2>::type>
inline Vector<T, N, true> operator*(const Vector<T, N, Ag1>& lhs,
                                    const Vector<T, N, Ag2>& rhs)
{
    auto r = internal::detach(lhs) * internal::detach(rhs);
    if (!internal::requires_grad(lhs) && !internal::requires_grad(rhs))
        return r;
    return Vector<T, N, true>(r, internal::MulBackward<T, N>{lhs, rhs});
}

template <typename T, std::size_t N, typename S,
          typename = typename std::enable_if<
              std::is_convertible<S, T>::value>::type>
inline Vector<T, N> operator*(const Vector<T, N>& v, S s)
{ return Vector<T, N>(v) *= s; }

template <typename T, std::size_t N, typename S,
          typename = typename std::enable_if<
              std::is_convertible<S, T>::value>::type>
inline Vector<T, N> operator*(S s, const Vector<T, N>& v)
{ return Vector<T, N>(v) *= s; }

template <typename T, std::size_t N, typename S,
          typename = typename std::enable_if<
              std::is_convertible<S, T>::value>::type>
inline Vector<T, N, true> operator*(Vector<T, N, true> v, S s)
{ return s * v; }

template <typename T, std::size_t N, typename S,
          typename = typename std::enable_if<
              std::is_convertible<S, T>::value>::type>
inline Vector<T, N, true> operator*(S s, Vector<T, N, true> v)
{
    Vector<T, N> r = s * v.detach();
    if (!v.requires_grad())
        return r;
    return Vector<T, N, true>(r, internal::ScalarMulBackward<T, N>{T(s), v});
}

template <typename T, std::size_t N>
inline Vector<T, N> operator/(const Vector<T, N>& lhs, const Vector<T, N>& rhs)
{ return Vector<T, N>(lhs) /= rhs; }

template <typename T, std::size_t N, bool Ag1, bool Ag2,
          typename = typename std::enable_if<Ag1 || Ag2>::type>
inline Vector<T, N, true> operator/(const Vector<T, N, Ag1>& lhs,
                                    const Vector<T, N, Ag2>& rhs)
{
    auto r = internal::detach(lhs) / internal::detach(rhs);
    if (!internal::requires_grad(lhs) && !internal::requires_grad(rhs))
        return r;
    return Vector<T, N, true>(r, internal::DivBackward<T, N>{lhs, rhs});
}

template <typename T, std::size_t N, typename S,
          typename = typename std::enable_if<std::is_convertible<S, T>::value>::type>
inline Vector<T, N> operator/(const Vector<T, N>& v, S s)
{ return Vector<T, N>(v) /= s; }

template <typename T, std::size_t N, typename S,
          typename = typename std::enable_if<
              std::is_convertible<S, T>::value>::type>
inline Vector<T, N, true> operator/(Vector<T, N, true> v, S s)
{
    Vector<T, N> r = v.detach() / s;
    if (!v.requires_grad())
        return r;
    return Vector<T, N, true>(r, internal::ScalarDivBackward<T, N>{v, T(s)});
}

template <typename T, std::size_t N, bool Ag>
inline std::ostream& operator<<(std::ostream& os, const Vector<T, N, Ag>& v)
{
    os << "Vector<" << typeid(T).name() << ", " << N;
    if (Ag)
        os << ", true";
    os << ">{";
    for (std::size_t i = 0; i < v.size()-1; ++i)
        os << v[i] << ", ";
    if (v.size() > 0)
        os << v[v.size()-1];
    return os << "}";
}

template <typename T, std::size_t N>
inline T dot(const Vector<T, N>& lhs, const Vector<T, N>& rhs)
{
    Vector<T, N> tmp = lhs * rhs;
    return std::accumulate(tmp.begin(), tmp.end(), T());
}

template <typename T, std::size_t N>
inline T norm(const Vector<T, N>& v)
{
    return sqrt(real(dot(v, v)));
}

// WARN: No checks for when the norm is zero!
template <typename T, std::size_t N>
inline Vector<T, N> normalize(const Vector<T, N>& v)
{
    return v / norm(v);
}

template <typename T>
inline Vector<T, 3> cross(const Vector<T, 3>& lhs, const Vector<T, 3>& rhs)
{
    Vector<T, 3> r;
    r[0] = lhs[1]*rhs[2] - lhs[2]*rhs[1];
    r[1] = lhs[2]*rhs[0] - lhs[0]*rhs[2];
    r[2] = lhs[0]*rhs[1] - lhs[1]*rhs[0];
    return r;
}

}
