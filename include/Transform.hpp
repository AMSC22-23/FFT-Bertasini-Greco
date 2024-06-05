/**
 * @file Transform.hpp
 * @brief Defines a template class Transform and its nested classes InputSpace and OutputSpace.
 */

#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP

#include <memory>

#include "typedefs.hpp"

namespace tr
{

    /**
     * @tparam T The data type to be used for input and output spaces.
     */
    template <typename T>
    class Transform
    {
    public:
        /**
         * @brief The InputSpace class represents the input space for the transformation.
         */
        class InputSpace
        {
        public:
            virtual ~InputSpace() = default;

            /**
             * @brief Get the data associated with the input space.
             * @return The data of the input space.
             */
            virtual auto get_data() const -> T = 0;
        };

        /**
         * @brief The OutputSpace class represents the output space for the transformation.
         */
        class OutputSpace
        {
        public:
            virtual ~OutputSpace() = default;

            /**
             * @brief Get the plottable representation of the output space.
             * @return The plottable representation of the output space.
             */
            virtual auto get_plottable_representation() const -> T = 0;

            /**
             * @brief Compress the output space data.
             * @param str The string representing the compression method.
             * @param val The compression value.
             */
            virtual auto compress(const std::string &str, const double val) -> void = 0;
        };

        virtual ~Transform() = default;

        /**
         * @brief Get the input space associated with the transformation.
         * @param data The data to be used for creating the input space.
         * @return A unique pointer to the InputSpace object.
         */
        virtual auto get_input_space(const T &data) const -> std::unique_ptr<Transform<T>::InputSpace> = 0;

        /**
         * @brief Get the output space associated with the transformation.
         * @return A unique pointer to the OutputSpace object.
         */
        virtual auto get_output_space() const -> std::unique_ptr<Transform<T>::OutputSpace> = 0;

        /**
         * @brief Perform the transformation operation.
         * @param input The input space for the transformation.
         * @param output The output space for the transformation.
         * @param flag A flag indicating additional transformation behavior.
         */
        virtual auto operator()(InputSpace &input, OutputSpace &output, const bool flag) const -> void = 0;
    };
}

#endif