#pragma once

namespace drt {

template <typename T>
class Autograd {
public:
    Autograd(T value)
      : m_value(value)
    { }

    virtual ~Autograd()
    { }

    T value()
    { return m_value; }

    virtual void backward(T weight) = 0;

protected:
    T m_value;
};

template <typename T>
class Variable : public Autograd<T> {
public:
    explicit Variable(T value)
      : Autograd<T>(value)
    { }

    T& value()
    { return this->m_value; }

    void backward(T weight) override
    { m_grad += weight; }

    T& grad()
    { return m_grad; }

private:
    T m_grad;
};

template <typename T>
class Constant : public Autograd<T> {
public:
    explicit Constant(T value)
      : Autograd<T>(value)
    { }

    T& value()
    { return this->m_value; }

    void backward(T weight) override
    { }
};

}
