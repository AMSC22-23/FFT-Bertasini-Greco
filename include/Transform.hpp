#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP

#include <typedefs.hpp>

class Transform{
public:
    class InputSpace
    {
    public:
        virtual ~InputSpace() = default;
        virtual auto get_data() const -> Typedefs::vec = 0;
    };

    class OutputSpace
    {
    public:
        virtual ~OutputSpace() = default;
        virtual auto get_plottable_representation() const -> Typedefs::vec = 0;
        virtual auto compress(const std::string&, const double) -> void = 0;
    };

    virtual ~Transform() = default;
    virtual auto get_input_space(Typedefs::vec &) const -> std::unique_ptr<Transform::InputSpace> = 0;
    virtual auto get_output_space() const -> std::unique_ptr<Transform::OutputSpace> = 0;
    virtual auto operator()(InputSpace&, OutputSpace&, bool) const -> void = 0;
};

#endif