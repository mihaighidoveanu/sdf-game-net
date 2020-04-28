from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef(""" 
    typedef struct Box{
        int x0,y0,x1,y1;
    }Box;
    char fontpath[500];
    void load_font(const char* path);
    int get_glyph_index(int codepoint);
    Box get_glyph_box(int glyph, double scale);
    double get_glyph_distance(int glyph, double scale, double x, double y);
""")


ffibuilder.set_source("_fontloader_cffi",
    """
    #include "fontloader.h"
    """,)

if __name__ == '__main__':
    ffibuilder.compile(verbose = True)
