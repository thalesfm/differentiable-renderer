#pragma once

#include <cmath>
#include <iostream>
#include <type_traits>

namespace drt {

template <typename T>
class Dual {
public:
    Dual() = default;

    Dual(const T& real, const T& dual = T()) : m_real(real), m_dual(dual) { }

    T& real()
    { return m_real; }

    const T& real() const
    { return m_real; }

    T& dual()
    { return m_dual; }

    const T& dual() const
    { return m_dual; }

    Dual& operator+=(const Dual& rhs)
    {
        m_real += rhs.real();
        m_dual += rhs.dual();
        return *this;
    }

    Dual& operator-=(const Dual& rhs)
    {
        m_real -= rhs.real();
        m_dual -= rhs.dual();
        return *this;
    }

    Dual& operator*=(const Dual& rhs)
    {
        T new_real = real() * rhs.real();
        T new_dual = real()*rhs.dual() + dual()*rhs.real();
        m_real = new_real;
        m_dual = new_dual;
        return *this;
    }

    // WARN: No checks for when rhs.real() is zero!
    Dual& operator/=(const Dual& rhs)
    {
        T new_real = real() / rhs.real();
        T new_dual = (dual()*rhs.real() - real()*rhs.dual()) /
            (rhs.real()*rhs.real());
        m_real = new_real;
        m_dual = new_dual;
        return *this;
    }

private:
    T m_real;
    T m_dual;
};

template <typename T>
inline Dual<T> operator+(Dual<T> lhs, const Dual<T>& rhs)
{ return lhs += rhs; }

template <typename T, typename S,
          typename = typename std::enable_if<
              std::is_convertible<S, T>::value>::type>
inline Dual<T> operator+(Dual<T> n, S s)
{ return n += Dual<T>(s); }

template <typename T, typename S,
          typename = typename std::enable_if<
              std::is_convertible<S, T>::value>::type>
inline Dual<T> operator+(S s, Dual<T> n)
{ return n += Dual<T>(s); }

template <typename T>
inline Dual<T> operator-(Dual<T> lhs, const Dual<T>& rhs)
{ return lhs -= rhs; }

template <typename T, typename S,
          typename = typename std::enable_if<
              std::is_convertible<S, T>::value>::type>
inline Dual<T> operator-(Dual<T> n, S s)
{ return n -= Dual<T>(s); }

template <typename T, typename S,
          typename = typename std::enable_if<
              std::is_convertible<S, T>::value>::type>
inline Dual<T> operator-(S s, const Dual<T>& n)
{ return Dual<T>(s) -= n; }

template <typename T>
inline Dual<T> operator*(Dual<T> lhs, const Dual<T>& rhs)
{ return lhs *= rhs; }

template <typename T, typename S,
          typename = typename std::enable_if<
              std::is_convertible<S, T>::value>::type>
inline Dual<T> operator*(Dual<T> n, S s)
{ return n *= Dual<T>(s); }

template <typename T, typename S,
          typename = typename std::enable_if<
              std::is_convertible<S, T>::value>::type>
inline Dual<T> operator*(S s, Dual<T> n)
{ return n *= Dual<T>(s); }

template <typename T>
inline Dual<T> operator/(Dual<T> lhs, const Dual<T>& rhs)
{ return lhs /= rhs; }

template <typename T, typename S,
          typename = typename std::enable_if<
              std::is_convertible<S, T>::value>::type>
inline Dual<T> operator/(Dual<T> n, S s)
{ return n /= Dual<T>(s); }

template <typename T, typename S,
          typename = typename std::enable_if<
              std::is_convertible<S, T>::value>::type>
inline Dual<T> operator/(S s, const Dual<T>& n)
{ return Dual<T>(s) /= n; }

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Dual<T>& n)
{ return os << n.real() << "+" << n.dual() << "e"; }

template <typename T>
inline T real(const Dual<T>& n)
{ return n.real(); }

template <typename T>
inline Dual<T> conj(const Dual<T>& n)
{ return n; }

template <typename T>
inline Dual<T> sqrt(const Dual<T>& n)
{
    T real = std::sqrt(n.real());
    T dual = n.dual() / (2*real);
    return Dual<T>(real, dual);
}

} // namespace drt

