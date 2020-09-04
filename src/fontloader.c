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

int load_font(const char* path){
    strcpy(fontpath, path);
    FILE *f = fopen(fontpath, "rb");
    if(f == NULL){
        return 0;
    }
    fread(ttf_buffer, 1, 1<<20, f);
    stbtt_InitFont(&font, ttf_buffer, stbtt_GetFontOffsetForIndex(ttf_buffer,0));
    fontInit = 1;
    fclose(f);
    return 1;
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
    printf("%s", fontpath);
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

int main(){
    load_font("../fonts/times.ttf");
    Box box = get_glyph_box(0, 20);
    printf("\n%d %d %d %d", box.x0, box.y0, box.x1, box.y1);
}
