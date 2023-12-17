# A few comments #

The code is simple but it works. You hve assumed unsigned int for the type of  the index. But normally the type for indexes is size_t. So I would use size_t instead of unsigned int. Of course then you need to parametrize your bit operations eith respect to the type of size_t, which is nomally 32 or 64 bits. But this is not a big deal.

C++20 has introduced std::biset to handle bitsets. 

Nice the idea of using matplotlib. Tye aliases are a good idea. But they should not pollute the global namespace. So I would put them in a namespace. Or include them into a struct.

I have put some notes in the code, they are marked with @note.


