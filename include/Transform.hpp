#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP

#include <memory>

#include "typedefs.hpp"

template <typename T>
class Transform{
public:
    class InputSpace
    {
    public:
        virtual ~InputSpace() = default;
        virtual auto get_data() const -> T = 0;
    };

    class OutputSpace
    {
    public:
        virtual ~OutputSpace() = default;
        virtual auto get_plottable_representation() const -> T = 0;
        virtual auto compress(const std::string&, const double) -> void = 0;
    };

    virtual ~Transform() = default;
    virtual auto get_input_space(const T &) const -> std::unique_ptr<Transform<T>::InputSpace> = 0;
    virtual auto get_output_space() const -> std::unique_ptr<Transform<T>::OutputSpace> = 0;
    virtual auto operator()(InputSpace&, OutputSpace&, const bool) const -> void = 0;
};

#endif