#include <stdlib.h>
#include <stdio.h>

#define STB_TRUETYPE_IMPLEMENTATION  // force following include to generate implementation
#define STBTT_STATIC
#include "ksv_truetype.h"

char fontpath[500];
unsigned char ttf_buffer[1<<20];
int fontInit = 0;
int glyphInit = 0, loadedGlyph = 0;
ksvtt_glyphsdfinfo glyphInfo;
stbtt_fontinfo font;

void load_font(const char* path){
    strcpy(fontpath, path);
    fread(ttf_buffer, 1, 1<<20, fopen(fontpath, "rb"));
    stbtt_InitFont(&font, ttf_buffer, stbtt_GetFontOffsetForIndex(ttf_buffer,0));
    fontInit = 1;
}


int get_glyph_index(int codepoint){
    if (!fontInit) {
        fontInit = 1;
        printf("You need to call load_font before any other font operation!");
        exit(1);
    }

    int index = stbtt_FindGlyphIndex(&font, codepoint);

    return index;
}

double get_glyph_distance(int glyph, double scale, double x, double y) {
    if (!fontInit) {
        fontInit = 1;
        printf("You need to call load_font before any other font operation!");
        exit(1);
    }
    
    float pixelScale = stbtt_ScaleForPixelHeight(&font, (float) scale);
    // float pixelScale = (float)scale;
    if (!glyphInit || loadedGlyph != glyph) {
        glyphInit = 1;
        ksvtt_GetGlyphSDFInfo(&font, &glyphInfo, glyph, pixelScale);
    }

    double distance = (double)stbtt_GetGlyphSignedDistance(&glyphInfo, pixelScale, (float)x, (float)y);
    
    return distance;
}

typedef struct Box{
    int x0,y0,x1,y1;
} Box;

Box get_glyph_box(int glyph, double scale) {
    if (!fontInit) {
        fontInit = 1;
        printf("You need to call load_font before any other font operation!");
        exit(1);
    }
    
    double pixelScale = (double)stbtt_ScaleForPixelHeight(&font, (float) scale);

    int x0, y0, x1, y1;
    stbtt_GetGlyphBox(&font, glyph, &x0, &y0, &x1, &y1);

    Box box = {.x0 = x0, .y0 = y0, .x1 = x1, .y1 = y1}; 

    return box;
}

