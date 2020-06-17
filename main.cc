#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>
#include <utility>
using std::swap;

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "glad/glad.h"
#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

constexpr int chunk_size = 16;
#define CHUNK_SIZE_STR "16"
#define BORDER_WIDTH_STR "0.1"

namespace {

constexpr float
    fovy_radians = 1.0f,
    near_plane = 0.3f,
    far_plane = 2048.0f,
    camera_speed = 4.f;

float raycast_distance_threshold = 120.f;

int screen_x = 1280, screen_y = 960;
SDL_Window* window = nullptr;
std::string argv0;
int chunks_to_draw = (unsigned)-1;
float current_fps = 120.0f;

static void panic(const char* message, const char* reason) {
    fprintf(stderr, "%s: %s %s\n", argv0.c_str(), message, reason);
    fflush(stderr);
    fflush(stdout);
    SDL_ShowSimpleMessageBox(
        SDL_MESSAGEBOX_ERROR, message, reason, nullptr
    );
    exit(1);
    abort();
}

#define PANIC_IF_GL_ERROR do { \
    if (GLenum PANIC_error = glGetError()) { \
        char PANIC_msg[160]; \
        snprintf(PANIC_msg, sizeof PANIC_msg, "line %i: code %u", __LINE__, (unsigned)PANIC_error); \
        panic("OpenGL error", PANIC_msg); \
    } \
} while (0)

static GLuint make_program(const char* vs_code, const char* fs_code)
{
    static GLchar log[1024];
    PANIC_IF_GL_ERROR;
    GLuint program_id = glCreateProgram();
    GLuint vs_id = glCreateShader(GL_VERTEX_SHADER);
    GLuint fs_id = glCreateShader(GL_FRAGMENT_SHADER);

    const GLchar* string_array[1];
    string_array[0] = (GLchar*)vs_code;
    glShaderSource(vs_id, 1, string_array, nullptr);
    string_array[0] = (GLchar*)fs_code;
    glShaderSource(fs_id, 1, string_array, nullptr);

    glCompileShader(vs_id);
    glCompileShader(fs_id);

    PANIC_IF_GL_ERROR;

    GLint okay = 0;
    GLsizei length = 0;
    const GLuint shader_id_array[2] = { vs_id, fs_id };
    for (auto id : shader_id_array) {
        glGetShaderiv(id, GL_COMPILE_STATUS, &okay);
        if (okay) {
            glAttachShader(program_id, id);
        } else {
            glGetShaderInfoLog(id, sizeof log, &length, log);
            fprintf(stderr, "%s\n", id == vs_id ? vs_code : fs_code);
            panic("Shader compilation error", log);
        }
    }

    glLinkProgram(program_id);
    glGetProgramiv(program_id, GL_LINK_STATUS, &okay);
    if (!okay) {
        glGetProgramInfoLog(program_id, sizeof log, &length, log);
        panic("Shader link error", log);
    }

    PANIC_IF_GL_ERROR;
    return program_id;
}

void load_cubemap_face(GLenum face, const char* filename)
{
    std::string full_filename = argv0 + "Tex/" + filename;
    SDL_Surface* surface = SDL_LoadBMP(full_filename.c_str());
    if (surface == nullptr) {
        panic(SDL_GetError(), full_filename.c_str());
    }
    if (surface->w != 1024 || surface->h != 1024) {
        panic("Expected 1024x1024 texture", full_filename.c_str());
    }
    if (surface->format->format != SDL_PIXELFORMAT_BGR24) {
        fprintf(stderr, "%i\n", (int)surface->format->format);
        panic("Expected 24-bit BGR bitmap", full_filename.c_str());
    }

    glTexImage2D(face, 0, GL_RGB, 1024, 1024, 0,
                  GL_BGR, GL_UNSIGNED_BYTE, surface->pixels);

    SDL_FreeSurface(surface);
}

GLuint load_cubemap()
{
    GLuint id = 0;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_CUBE_MAP, id);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_LOD, 0);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LOD, 8);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, 8);

    load_cubemap_face(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, "left.bmp");
    load_cubemap_face(GL_TEXTURE_CUBE_MAP_POSITIVE_X, "right.bmp");
    load_cubemap_face(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, "bottom.bmp");
    load_cubemap_face(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, "top.bmp");
    load_cubemap_face(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, "back.bmp");
    load_cubemap_face(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, "front.bmp");

    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    PANIC_IF_GL_ERROR;
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    return id;
}

static const char skybox_vs_source[] =
"#version 330\n"
"layout(location=0) in vec3 position;\n"
"out vec3 texture_coordinate;\n"
"uniform mat4 view_matrix;\n"
"uniform mat4 proj_matrix;\n"
"void main() {\n"
    "vec4 v = view_matrix * vec4(400*position, 0.0);\n"
    "gl_Position = proj_matrix * vec4(v.xyz, 1);\n"
    "texture_coordinate = position;\n"
"}\n";

static const char skybox_fs_source[] =
"#version 330\n"
"in vec3 texture_coordinate;\n"
"out vec4 color;\n"
"uniform samplerCube cubemap;\n"
"void main() {\n"
    "vec4 c = texture(cubemap, texture_coordinate);\n"
    "c.a = 1.0;\n"
    "color = c;\n"
    "gl_FragDepth = 0.99999;\n"
"}\n";

static const float skybox_vertices[24] = {
    -1, 1, 1,
    -1, -1, 1,
    1, -1, 1,
    1, 1, 1,
    -1, 1, -1,
    -1, -1, -1,
    1, -1, -1,
    1, 1, -1,
};

static const GLushort skybox_elements[36] = {
    7, 4, 5, 7, 5, 6,
    1, 0, 3, 1, 3, 2,
    5, 1, 2, 5, 2, 6,
    4, 7, 3, 4, 3, 0,
    0, 1, 5, 0, 5, 4,
    2, 3, 7, 2, 7, 6
};

void draw_skybox(
    glm::mat4 view_matrix, glm::mat4 proj_matrix)
{
    static bool cubemap_loaded = false;
    static GLuint cubemap_texture_id;
    if (!cubemap_loaded) {
        cubemap_texture_id = load_cubemap();
        cubemap_loaded = true;
    }

    static GLuint vao = 0;
    static GLuint program_id;
    static GLuint vertex_buffer_id;
    static GLuint element_buffer_id;
    static GLint view_matrix_id;
    static GLint proj_matrix_id;
    static GLint cubemap_uniform_id;

    if (vao == 0) {
        program_id = make_program(skybox_vs_source, skybox_fs_source);
        view_matrix_id = glGetUniformLocation(program_id, "view_matrix");
        proj_matrix_id = glGetUniformLocation(program_id, "proj_matrix");
        cubemap_uniform_id = glGetUniformLocation(program_id, "cubemap");

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vertex_buffer_id);
        glGenBuffers(1, &element_buffer_id);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_id);
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER, sizeof skybox_elements,
            skybox_elements, GL_STATIC_DRAW
        );

        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id);
        glBufferData(
            GL_ARRAY_BUFFER, sizeof skybox_vertices,
            skybox_vertices, GL_STATIC_DRAW
        );
        glVertexAttribPointer(
            0,
            3,
            GL_FLOAT,
            false,
            sizeof(float) * 3,
            (void*)0
        );
        glEnableVertexAttribArray(0);
        PANIC_IF_GL_ERROR;
    }

    glUseProgram(program_id);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture_id);
    glUniform1i(cubemap_uniform_id, 0);

    glUniformMatrix4fv(view_matrix_id, 1, 0, &view_matrix[0][0]);
    glUniformMatrix4fv(proj_matrix_id, 1, 0, &proj_matrix[0][0]);

    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, (void*)0);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    PANIC_IF_GL_ERROR;
}

class chunk
{
    friend void draw_chunk_raycast(
        chunk& c, glm::vec3 eye, glm::mat4 vp_matrix, bool first_time);
    friend void draw_chunk_conventional(
        chunk& c, glm::mat4 vp_matrix, bool first_time);
    friend void update_window_title(glm::vec3);

    static int loaded_vbo;

    bool dirty = false;
    uint16_t blocks[chunk_size][chunk_size][chunk_size] = {};
    GLuint vertex_buffer_id = 0;
    unsigned vertex_count = 0;
    GLuint texture_name = 0;
    int32_t opaque_block_count = 0;
    int16_t x_block_counts[chunk_size] = { 0 };
    int16_t y_block_counts[chunk_size] = { 0 };
    int16_t z_block_counts[chunk_size] = { 0 };
    glm::vec3 aabb_low = glm::vec3(0,0,0);
    glm::vec3 aabb_high = glm::vec3(0,0,0);

    void fix_dirty()
    {
        if (!dirty) return;

        aabb_low = glm::vec3(chunk_size, chunk_size, chunk_size);
        aabb_high = glm::vec3(0, 0, 0);
        int32_t x_counts = 0, y_counts = 0, z_counts = 0;
        for (size_t i = 0; i < chunk_size; ++i) {
            x_counts += x_block_counts[i];
            y_counts += y_block_counts[i];
            z_counts += z_block_counts[i];
            if (x_block_counts[i] >= 1) {
                aabb_low.x = std::min<float>(i, aabb_low.x);
                aabb_high.x = std::max<float>(i+1, aabb_high.x);
            }
            if (y_block_counts[i] >= 1) {
                aabb_low.y = std::min<float>(i, aabb_low.y);
                aabb_high.y = std::max<float>(i+1, aabb_high.y);
            }
            if (z_block_counts[i] >= 1) {
                aabb_low.z = std::min<float>(i, aabb_low.z);
                aabb_high.z = std::max<float>(i+1, aabb_high.z);
            }
        }

        assert(x_counts == opaque_block_count);
        assert(y_counts == opaque_block_count);
        assert(z_counts == opaque_block_count);

        if (texture_name != 0) {
            glBindTexture(GL_TEXTURE_3D, texture_name);
            glTexSubImage3D(GL_TEXTURE_3D, 0,
                            0, 0, 0, chunk_size, chunk_size, chunk_size,
                            GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, blocks);
        }
        fill_vertex_buffer();

        dirty = false;
        PANIC_IF_GL_ERROR;
    }
  public:
    const glm::vec3 position;

    chunk(glm::vec3 in_position) : position(in_position)
    {

    }

  private:
    GLuint get_texture_name()
    {
        if (texture_name == 0) {
            glGenTextures(1, &texture_name);
            glBindTexture(GL_TEXTURE_3D, texture_name);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            PANIC_IF_GL_ERROR;
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA,
                         chunk_size, chunk_size, chunk_size, 0,
                         GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, blocks);
            PANIC_IF_GL_ERROR;
        }
        fix_dirty();
        return texture_name;
    }

    struct vertex
    {
        float x, y, z;
        uint16_t color;
        vertex(float x_, float y_, float z_, uint16_t c) :
            x(x_), y(y_), z(z_), color(c) { }
    };

    GLuint get_vertex_buffer_id(unsigned* out_vertex_count)
    {
        fix_dirty();
        if (vertex_buffer_id == 0) {
            ++loaded_vbo;
            glGenBuffers(1, &vertex_buffer_id);
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id);
            fill_vertex_buffer();
        }
        *out_vertex_count = vertex_count;
        return vertex_buffer_id;
    }

    void fill_vertex_buffer()
    {
        if (vertex_buffer_id == 0) return;

        auto opaque_block = [&] (int x, int y, int z) -> bool
        {
            // TODO properly handle edge cases with other chunks.
            if (x < 0 or x >= chunk_size
             or y < 0 or y >= chunk_size
             or z < 0 or z >= chunk_size) return false;

            return blocks[z][y][x] & 1;
        };

        vertex_count = 0;
        std::vector<vertex> verts;
        auto verts_add_block = [&] (int x, int y, int z)
        {
            uint16_t c = blocks[z][y][x];
            if ((c & 1) == 0) return;
            float x1 = x+1;
            float y1 = y+1;
            float z1 = z+1;

            if (!opaque_block(x-1, y, z)) {
                verts.emplace_back(x, y1, z1, c);
                verts.emplace_back(x, y1, z, c);
                verts.emplace_back(x, y, z1, c);
                verts.emplace_back(x, y1, z, c);
                verts.emplace_back(x, y, z, c);
                verts.emplace_back(x, y, z1, c);
            }
            if (!opaque_block(x1, y, z)) {
                verts.emplace_back(x1, y, z, c);
                verts.emplace_back(x1, y1, z, c);
                verts.emplace_back(x1, y, z1, c);
                verts.emplace_back(x1, y1, z, c);
                verts.emplace_back(x1, y1, z1, c);
                verts.emplace_back(x1, y, z1, c);
            }
            if (!opaque_block(x, y-1, z)) {
                verts.emplace_back(x, y, z, c);
                verts.emplace_back(x1, y, z1, c);
                verts.emplace_back(x, y, z1, c);
                verts.emplace_back(x, y, z, c);
                verts.emplace_back(x1, y, z, c);
                verts.emplace_back(x1, y, z1, c);
            }
            if (!opaque_block(x, y1, z)) {
                verts.emplace_back(x1, y1, z1, c);
                verts.emplace_back(x1, y1, z, c);
                verts.emplace_back(x, y1, z, c);
                verts.emplace_back(x, y1, z1, c);
                verts.emplace_back(x1, y1, z1, c);
                verts.emplace_back(x, y1, z, c);
            }
            if (!opaque_block(x, y, z-1)) {
                verts.emplace_back(x, y, z, c);
                verts.emplace_back(x, y1, z, c);
                verts.emplace_back(x1, y, z, c);
                verts.emplace_back(x1, y1, z, c);
                verts.emplace_back(x1, y, z, c);
                verts.emplace_back(x, y1, z, c);
            }
            if (!opaque_block(x, y, z1)) {
                verts.emplace_back(x, y1, z1, c);
                verts.emplace_back(x1, y, z1, c);
                verts.emplace_back(x1, y1, z1, c);
                verts.emplace_back(x1, y, z1, c);
                verts.emplace_back(x, y1, z1, c);
                verts.emplace_back(x, y, z1, c);
            }
        };

        for (int z = 0; z < chunk_size; ++z) {
            for (int y = 0; y < chunk_size; ++y) {
                for (int x = 0; x < chunk_size; ++x) {
                    verts_add_block(x,y,z);
                }
            }
        }

        vertex_count = unsigned(verts.size());
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id);
        glBufferData(
            GL_ARRAY_BUFFER,
            vertex_count * (sizeof verts[0]),
            verts.data(),
            GL_STATIC_DRAW);
    }

  public:
    void unload_texture()
    {
        if (texture_name != 0) {
            glDeleteTextures(1, &texture_name);
            texture_name = 0;
            PANIC_IF_GL_ERROR;
        }
    }

    void unload_vertex_buffer()
    {
        if (vertex_buffer_id != 0) {
            glDeleteBuffers(1, &vertex_buffer_id);
            vertex_buffer_id = 0;
            --loaded_vbo;
        }
    }

    int32_t get_opaque_block_count() const
    {
        return opaque_block_count;
    }

    void set_block(size_t x, size_t y, size_t z, uint16_t block)
    {
        dirty = true;
        assert(x < chunk_size);
        assert(y < chunk_size);
        assert(z < chunk_size);
        const int old_alpha = int(blocks[z][y][x] & 1);
        const int new_alpha = int(block & 1);
        blocks[z][y][x] = block;
        const int16_t block_count_change = int16_t(new_alpha - old_alpha);
        assert(block_count_change <= 1 && block_count_change >= -1);
        opaque_block_count += block_count_change;
        x_block_counts[x] += block_count_change;
        y_block_counts[y] += block_count_change;
        z_block_counts[z] += block_count_change;
    }

    bool in_chunk(glm::vec3 v) const
    {
        return position.x < v.x && v.x < position.x + chunk_size &&
               position.y < v.y && v.y < position.y + chunk_size &&
               position.z < v.z && v.z < position.z + chunk_size;
    }

    ~chunk()
    {
        unload_texture();
        unload_vertex_buffer();
    }

    chunk(chunk&&) = delete;
};

int chunk::loaded_vbo = 0;
bool chunk_debug = false;

static const float chunk_vertices[32] =
{
    0, 1, 1, 1,
    0, 0, 1, 1,
    1, 0, 1, 1,
    1, 1, 1, 1,
    0, 1, 0, 1,
    0, 0, 0, 1,
    1, 0, 0, 1,
    1, 1, 0, 1,
};

static const GLushort chunk_elements[36] = {
    6, 7, 2, 7, 3, 2,
    4, 5, 0, 5, 1, 0,
    0, 3, 4, 3, 7, 4,
    6, 2, 5, 2, 1, 5,
    2, 3, 1, 3, 0, 1,
    6, 5, 7, 5, 4, 7,
};

static const char chunk_vs_source[] =
"#version 330\n"
"layout(location=0) in vec4 in_vertex;\n"
"out vec4 unit_box_position;\n"
"out float border_fade;\n"
"uniform vec3 chunk_offset;\n"
"uniform mat4 vp_matrix;\n"
"uniform vec3 aabb_low;\n"
"uniform vec3 aabb_size;\n"
"uniform vec3 eye_in_model_space;\n"
"void main() {\n"
    "vec4 model_space_pos = vec4(in_vertex.xyz * aabb_size + aabb_low, 1);\n"
    "gl_Position = vp_matrix * \n"
        "(model_space_pos + vec4(chunk_offset, 0));\n"
    "unit_box_position = in_vertex;\n"
    "vec3 disp = model_space_pos.xyz - eye_in_model_space;\n"
    "float distance = sqrt(dot(disp, disp));\n"
    "border_fade = clamp(sqrt(distance) * 0.042, 0.5, 1.0);\n"
"}\n";

static const char chunk_fs_source[] =
"#version 330\n"
"in vec4 unit_box_position;\n"
"in float border_fade;\n"
"uniform vec3 eye_in_model_space;\n"
"uniform sampler3D chunk_blocks;\n"
"uniform bool chunk_debug;\n"
"uniform vec3 aabb_low;\n"
"uniform vec3 aabb_size;\n"
"out vec4 color;\n"
"void main() {\n"
// Needs to be calculated in fs, not vs, due to interpolation rounding errors.
"vec4 model_space_position = vec4(unit_box_position.xyz\n"
                          "* aabb_size + aabb_low, 1);\n"
"if (!chunk_debug) {\n"
    "const float d = " BORDER_WIDTH_STR ";\n"
    "float x0 = eye_in_model_space.x;\n"
    "float y0 = eye_in_model_space.y;\n"
    "float z0 = eye_in_model_space.z;\n"
    "vec3 slope = vec3(model_space_position) - eye_in_model_space;\n"
    "float xm = slope.x;\n"
    "float ym = slope.y;\n"
    "float zm = slope.z;\n"
    "float rcp = 1.0/" CHUNK_SIZE_STR ".0;\n"
    "float best_t = 1.0 / 0.0;\n"
    "vec4 best_color = vec4(0,0,0,0);\n"
    "vec3 best_coord = vec3(0,0,0);\n"
    //"int x_init = xm > 0 ? 0 : " CHUNK_SIZE_STR ";\n"
    "int x_init = int(xm > 0 ? ceil(model_space_position.x) \n"
               ": floor(model_space_position.x));\n"
    "int x_end = xm > 0 ? " CHUNK_SIZE_STR " : 0;\n"
    "int x_step = xm > 0 ? 1 : -1;\n"
    "float x_fudge = xm > 0 ? .25 : -.25;\n"
    "for (int x = x_init; x != x_end; x += x_step) {\n"
        "float t = (x - x0) / xm;\n"
        "float y = y0 + ym * t;\n"
        "float z = z0 + zm * t;\n"
        "if (y < 0 || y > " CHUNK_SIZE_STR ") break;\n"
        "if (z < 0 || z > " CHUNK_SIZE_STR ") break;\n"
        "vec3 texcoord = vec3(x + x_fudge, y, z) * rcp;\n"
        "vec4 lookup_color = texture(chunk_blocks, texcoord);\n"
        "if (lookup_color.a > 0 && t > 0) {\n"
            "if (best_t > t) {\n"
                "best_t = t;\n"
                "best_color = lookup_color;\n"
                "best_coord = vec3(x,y,z);\n"
                "if (y - floor(y + d) < d || z - floor(z + d) < d) {\n"
                    "best_color.rgb *= border_fade;\n"
                "}\n"
            "}\n"
            "break;\n"
        "}\n"
    "}\n"
    "int y_init = int(ym > 0 ? ceil(model_space_position.y) \n"
               ": floor(model_space_position.y));\n"
    "int y_end = ym > 0 ? " CHUNK_SIZE_STR " : 0;\n"
    "int y_step = ym > 0 ? 1 : -1;\n"
    "float y_fudge = ym > 0 ? .05 : -.05;\n"
    "for (int y = y_init; y != y_end; y += y_step) {\n"
        "float t = (y - y0) / ym;\n"
        "float x = x0 + xm * t;\n"
        "float z = z0 + zm * t;\n"
        "if (x < 0 || x > " CHUNK_SIZE_STR ") break;\n"
        "if (z < 0 || z > " CHUNK_SIZE_STR ") break;\n"
        "vec3 texcoord = vec3(x, y + y_fudge, z) * rcp;\n"
        "vec4 lookup_color = texture(chunk_blocks, texcoord);\n"
        "if (lookup_color.a > 0 && t > 0) {\n"
            "if (best_t > t) {\n"
                "best_t = t;\n"
                "best_color = lookup_color;\n"
                "best_coord = vec3(x,y,z);\n"
                "if (x - floor(x + d) < d || z - floor(z + d) < d) {\n"
                    "best_color.rgb *= border_fade;\n"
                "}\n"
            "}\n"
            "break;\n"
        "}\n"
    "}\n"
    "int z_init = int(zm > 0 ? ceil(model_space_position.z) \n"
               ": floor(model_space_position.z));\n"
    "int z_end = zm > 0 ? " CHUNK_SIZE_STR " : 0;\n"
    "int z_step = zm > 0 ? 1 : -1;\n"
    "float z_fudge = zm > 0 ? .05 : -.05;\n"
    "for (int z = z_init; z != z_end; z += z_step) {\n"
        "float t = (z - z0) / zm;\n"
        "float x = x0 + xm * t;\n"
        "float y = y0 + ym * t;\n"
        "if (x < 0 || x > " CHUNK_SIZE_STR ") break;\n"
        "if (y < 0 || y > " CHUNK_SIZE_STR ") break;\n"
        "vec3 texcoord = vec3(x, y, z + z_fudge) * rcp;\n"
        "vec4 lookup_color = texture(chunk_blocks, texcoord);\n"
        "if (lookup_color.a > 0 && t > 0) {\n"
            "if (best_t > t) {\n"
                "best_t = t;\n"
                "best_color = lookup_color;\n"
                "best_coord = vec3(x,y,z);\n"
                "if (x - floor(x + d) < d || y - floor(y + d) < d) {\n"
                    "best_color.rgb *= border_fade;\n"
                "}\n"
            "}\n"
            "break;\n"
        "}\n"
    "}\n"
    "if (best_color.a == 0) discard;\n"
    "color = best_color;\n"
    //"vec4 v = vp_matrix * vec4(best_coord + chunk_offset, 1);\n"
    //"gl_FragDepth = .3;\n"
"} else {\n"
    // "color = texture(chunk_blocks, model_space_position.xyz / 16.0);\n"
    "int x_floor = int(floor(model_space_position.x));\n"
    "int y_floor = int(floor(model_space_position.y));\n"
    "int z_floor = int(floor(model_space_position.z));\n"
    "color = vec4(x_floor & 1, y_floor & 1, z_floor & 1, 1);\n"
"}\n"
"}\n";

void draw_chunk_raycast(
    chunk& c, glm::vec3 eye, glm::mat4 vp_matrix, bool first_time)
{
    static GLuint vao = 0;
    static GLuint program_id;
    static GLuint vertex_buffer_id;
    static GLuint element_buffer_id;
    static GLint chunk_offset_id;
    static GLint vp_matrix_id;
    static GLint aabb_low_id;
    static GLint aabb_size_id;
    static GLint chunk_blocks_id;
    static GLint eye_in_model_space_id;
    static GLint chunk_debug_id;

    if (vao == 0) {
        program_id = make_program(chunk_vs_source, chunk_fs_source);
        chunk_offset_id = glGetUniformLocation(program_id, "chunk_offset");
        vp_matrix_id = glGetUniformLocation(program_id, "vp_matrix");
        aabb_low_id = glGetUniformLocation(program_id, "aabb_low");
        aabb_size_id = glGetUniformLocation(program_id, "aabb_size");
        chunk_blocks_id = glGetUniformLocation(program_id, "chunk_blocks");
        eye_in_model_space_id = glGetUniformLocation(
            program_id, "eye_in_model_space");
        chunk_debug_id = glGetUniformLocation(program_id, "chunk_debug");

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vertex_buffer_id);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id);
        glBufferData(
            GL_ARRAY_BUFFER, sizeof chunk_vertices,
            chunk_vertices, GL_STATIC_DRAW);

        glGenBuffers(1, &element_buffer_id);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_id);
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER, sizeof chunk_elements,
            chunk_elements, GL_STATIC_DRAW);

        glVertexAttribPointer(
            0,
            4,
            GL_FLOAT,
            false,
            sizeof(float) * 4,
            (void*)0);
        glEnableVertexAttribArray(0);
        PANIC_IF_GL_ERROR;
    }

    glm::vec3 eye_in_model_space = (eye - glm::vec3(c.position));

    if (first_time) {
        PANIC_IF_GL_ERROR;
        glUseProgram(program_id);
        glBindVertexArray(vao);
        glUniformMatrix4fv(vp_matrix_id, 1, 0, &vp_matrix[0][0]);
        glUniform1i(chunk_debug_id, chunk_debug);
        glActiveTexture(GL_TEXTURE0);
        glUniform1i(chunk_blocks_id, 0);
    }

    if (c.get_opaque_block_count() == 0) {
        return;
    }

    glBindTexture(GL_TEXTURE_3D, c.get_texture_name());
    PANIC_IF_GL_ERROR;

    glUniform3fv(chunk_offset_id, 1, &c.position[0]);
    glUniform3fv(aabb_low_id, 1, &c.aabb_low[0]);
    glUniform3fv(aabb_size_id, 1, &(c.aabb_high - c.aabb_low)[0]);
    glUniform3fv(eye_in_model_space_id, 1, &eye_in_model_space[0]);
    PANIC_IF_GL_ERROR;

    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, (void*)0);
    PANIC_IF_GL_ERROR;
}

const char chunk_conventional_vs_source[] =
"#version 330\n"
"layout(location=0) in vec3 model_space_position;\n"
"layout(location=1) in int integer_color;\n"
"out vec3 color;\n"
"out vec3 model_space_position_;\n"
"uniform vec3 chunk_offset;\n"
"uniform mat4 vp_matrix;\n"
"void main() {\n"
    "vec4 v = vec4(chunk_offset + model_space_position, 1);\n"
    "gl_Position = vp_matrix * v;\n"
    "float red   = ((integer_color >> 11) & 31) * (1./31.);\n"
    "float green = ((integer_color >> 6) & 31) * (1./31.);\n"
    "float blue  = ((integer_color >> 1) & 31) * (1./31.);\n"
    "color = vec3(red, green, blue);\n"
    "model_space_position_ = model_space_position;\n"
"}\n";

const char chunk_conventional_fs_source[] =
"#version 330\n"
"in vec3 color;\n"
"in vec3 model_space_position_;\n"
"out vec4 out_color;\n"
"void main() {\n"
    "const float d = " BORDER_WIDTH_STR ";\n"
    "float x = model_space_position_.x;\n"
    "int x_border = (x - floor(x + d) < d) ? 1 : 0;\n"
    "float y = model_space_position_.y;\n"
    "int y_border = (y - floor(y + d) < d) ? 1 : 0;\n"
    "float z = model_space_position_.z;\n"
    "int z_border = (z - floor(z + d) < d) ? 1 : 0;\n"
    "float scale = (x_border + y_border + z_border >= 2) ? 0.5 : 1.0;\n"
    "out_color = vec4(color * scale, 1);\n"
"}\n";

void draw_chunk_conventional(
    chunk& c, glm::mat4 vp_matrix, bool first_time)
{
    static GLuint vao = 0;
    static GLuint program_id;
    static GLint chunk_offset_id;
    static GLint vp_matrix_id;
    if (vao == 0) {
        program_id = make_program(chunk_conventional_vs_source,
                                  chunk_conventional_fs_source);
        chunk_offset_id = glGetUniformLocation(program_id, "chunk_offset");
        vp_matrix_id = glGetUniformLocation(program_id, "vp_matrix");
        PANIC_IF_GL_ERROR;

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
    }

    if (first_time) {
        glBindVertexArray(vao);
        glUseProgram(program_id);
        PANIC_IF_GL_ERROR;

        glUniformMatrix4fv(vp_matrix_id, 1, 0, &vp_matrix[0][0]);
        PANIC_IF_GL_ERROR;
    }

    if (c.get_opaque_block_count() == 0) {
        return;
    }

    glUniform3fv(chunk_offset_id, 1, &c.position[0]);
    unsigned vertex_count;
    auto vbo_id = c.get_vertex_buffer_id(&vertex_count);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id);
    assert(vbo_id != 0);

    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        false,
        sizeof (chunk::vertex),
        (void*)offsetof(chunk::vertex, x));
    glEnableVertexAttribArray(0);
    PANIC_IF_GL_ERROR;

    glVertexAttribIPointer(
        1,
        1,
        GL_UNSIGNED_SHORT,
        sizeof (chunk::vertex),
        (void*)offsetof(chunk::vertex, color));
    glEnableVertexAttribArray(1);
    PANIC_IF_GL_ERROR;

    glDrawArrays(GL_TRIANGLES, 0, vertex_count);
    PANIC_IF_GL_ERROR;
}

class world
{
  public: // XXX
    std::unordered_map<uint64_t, std::unique_ptr<chunk>> chunk_map;
  public:
    chunk* chunk_ptr(int x_chunk, int y_chunk, int z_chunk, bool needed=false)
    {
        assert(int16_t(x_chunk) == x_chunk);
        assert(int16_t(y_chunk) == y_chunk);
        assert(int16_t(z_chunk) == z_chunk);
        uint64_t key = uint16_t(x_chunk);
        key = key << 16 | uint16_t(y_chunk);
        key = key << 16 | uint16_t(z_chunk);
        if (needed) {
            std::unique_ptr<chunk>& ptr = chunk_map[key];
            if (ptr == nullptr) {
                ptr = std::make_unique<chunk>(glm::vec3(
                    x_chunk * chunk_size,
                    y_chunk * chunk_size,
                    z_chunk * chunk_size));
            }
            return ptr.get();
        }
        auto iter = chunk_map.find(key);
        return iter == chunk_map.end() ? nullptr : iter->second.get();
    }

    chunk& chunk_ref(int x_chunk, int y_chunk, int z_chunk)
    {
        chunk* ptr = chunk_ptr(x_chunk, y_chunk, z_chunk, true);
        assert(ptr);
        return *ptr;
    }

    void set_block(int x, int y, int z, uint16_t block)
    {
        unsigned x_in_chunk = unsigned(x) % chunk_size;
        unsigned y_in_chunk = unsigned(y) % chunk_size;
        unsigned z_in_chunk = unsigned(z) % chunk_size;
        int x_chunk = (x - int(x_in_chunk)) / chunk_size;
        int y_chunk = (y - int(y_in_chunk)) / chunk_size;
        int z_chunk = (z - int(z_in_chunk)) / chunk_size;
        assert(x_chunk * chunk_size + int(x_in_chunk) == x);
        chunk_ref(x_chunk, y_chunk, z_chunk).set_block(
            x_in_chunk, y_in_chunk, z_in_chunk, block);
    }
};

world the_world;

void draw_scene(
    glm::vec3 eye,
    glm::vec3 forward_normal_vector,
    glm::mat4 view_matrix,
    glm::mat4 proj_matrix)
{
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    auto vp_matrix = proj_matrix * view_matrix;
    draw_skybox(view_matrix, proj_matrix);

    // Sort from nearest to furthest (reverse painters)
    std::vector<std::pair<float, chunk*>> raycast_chunks_by_depth;
    std::unordered_set<chunk*> nearby_chunks;

    float h = chunk_size * 0.5f;
    auto fixup =
        glm::dot(forward_normal_vector, glm::vec3(h,h,h) - eye);

    unsigned count = 0;
    for (auto& key_chunk_ptr_pair : the_world.chunk_map) {
        chunk& chunk_to_draw = *key_chunk_ptr_pair.second;
        float depth = glm::dot(forward_normal_vector, chunk_to_draw.position)
                    + fixup;
        if (depth < -chunk_size) {
            continue;
        }
        glm::vec3 disp = chunk_to_draw.position - eye;
        float squared_dist = glm::dot(disp, disp);
        float squared_thresh = raycast_distance_threshold
                                       * raycast_distance_threshold;
        if (squared_dist >= squared_thresh) {
            raycast_chunks_by_depth.emplace_back(depth, &chunk_to_draw);
        }
        else {
            nearby_chunks.insert(&chunk_to_draw);
        }
    }
    std::sort(raycast_chunks_by_depth.begin(), raycast_chunks_by_depth.end());

    bool first_time = true;
    for (chunk* chunk_ptr : nearby_chunks) {
        if (count++ >= unsigned(chunks_to_draw)) break;
        draw_chunk_conventional(*chunk_ptr, vp_matrix, first_time);
        first_time = false;
    }

    first_time = true;
    for (auto pair : raycast_chunks_by_depth)
    {
        if (count++ >= unsigned(chunks_to_draw)) break;
        draw_chunk_raycast(*pair.second, eye, vp_matrix, first_time);
        first_time = false;
    }

    // Save gpu memory
    for (auto& pair : the_world.chunk_map) {
        chunk* chunk_ptr = pair.second.get();
        if (nearby_chunks.count(chunk_ptr) == 0) {
            chunk_ptr->unload_vertex_buffer();
        }
    }
}

template <int N, int Border>
class congestion_model
{
    constexpr static uint16_t
        red_category = 1,
        green_category = 2,
        blue_category = 3,
        blank = 0;

    struct tile
    {
        int red_next_ptrdiff = 0;
        int green_next_ptrdiff = 0;
        int blue_next_ptrdiff = 0;
        int x = 0, y = 0, z = 0;
        uint16_t old_category = 0, current_category = 0;
        uint16_t color = 0;
    };
    std::array<tile, N*N*6> tile_array;
    uint16_t tick_color = blue_category;

  public:
    congestion_model(std::mt19937& rng, double probability)
    {
        tile* pos_x_face = &tile_array[N*N*0];
        tile* pos_y_face = &tile_array[N*N*1];
        tile* pos_z_face = &tile_array[N*N*2];
        tile* neg_x_face = &tile_array[N*N*3];
        tile* neg_y_face = &tile_array[N*N*4];
        tile* neg_z_face = &tile_array[N*N*5];

        tile* red_next = nullptr;
        tile* green_next = nullptr;
        tile* blue_next = nullptr;

        uint32_t thresh = uint32_t(2147483648.0 * probability);
        auto maybe_fill_tile = [&] (tile& t, uint16_t c0, uint16_t c1)
        {
            assert(t.current_category == blank);
            uint32_t r = rng();
            uint16_t set_category = blank;
            if (r < thresh) {
                t.current_category = c0;
                set_category = c0;
            }
            if (r > ~thresh) {
                t.current_category = c1;
                set_category = c1;
            }
            uint16_t color = 0;
            if (set_category != blank) {
                auto bump0 = rng() % 9;
                auto bump1 = rng() % 5;
                if (bump0 <= 1 && bump1 <= 0) {
                    switch (set_category) {
                        default: assert(0);
                        break; case red_category:
                            color = 31 << 11 | 17 << 6 | 20 << 1 | 1;
                        break; case green_category:
                            color = 31 << 11 | 31 << 6 | 7 << 1 | 1;
                        break; case blue_category:
                            color = 15 << 6 | 31 << 1 | 1;
                    }
                }
                else {
                    switch (set_category) {
                        default: assert(0);
                        break; case red_category:
                            color = (16+bump0) << 11 | 4 << 6
                                  | (7+bump1) << 1 | 1;
                        break; case green_category:
                            color = (bump1 + 8) << 11
                                  | (28-bump0) << 6 | 1;
                        break; case blue_category:
                            color = (19+bump0) << 11 | (19+bump0) << 6
                                  | (31-bump1) << 1 | 1;
                    }
                }
            }
            the_world.set_block(t.x, t.y, t.z, color);
            t.color = color;
        };

        for (int y = 0; y < N; ++y) {
            for (int z = 0; z < N; ++z) {
                tile& t = neg_x_face[y*N + z];

                green_next = z == 0 ?
                            &neg_z_face[0*N + y] : &neg_x_face[y*N + z-1];
                blue_next = y == 0 ?
                            &neg_y_face[0*N + z] : &neg_x_face[(y-1)*N + z];
                t.green_next_ptrdiff = int(green_next - &t);
                t.blue_next_ptrdiff  = int(blue_next  - &t);
                t.x = -1;
                t.y = y;
                t.z = z;
                if (Border <= y and y < N - Border
                and Border <= z and z < N - Border) {
                    maybe_fill_tile(t, green_category, blue_category);
                }
            }
        }

        for (int y = 0; y < N; ++y) {
            for (int z = 0; z < N; ++z) {
                tile& t = pos_x_face[y*N + z];

                green_next = z == N-1 ?
                            &pos_z_face[(N-1)*N + y] : &pos_x_face[y*N + z+1];
                blue_next = y == N-1 ?
                            &pos_y_face[(N-1)*N + z] : &pos_x_face[(y+1)*N + z];
                t.green_next_ptrdiff = int(green_next - &t);
                t.blue_next_ptrdiff  = int(blue_next  - &t);
                t.x = N;
                t.y = y;
                t.z = z;
                if (Border <= y and y < N - Border
                and Border <= z and z < N - Border) {
                    maybe_fill_tile(t, green_category, blue_category);
                }
            }
        }

        for (int x = 0; x < N; ++x) {
            for (int z = 0; z < N; ++z) {
                tile& t = neg_y_face[x*N + z];

                red_next = z == 0 ?
                          &neg_z_face[x*N + 0] : &neg_y_face[x*N + z-1];
                blue_next = x == N-1 ?
                          &pos_x_face[0*N + z] : &neg_y_face[(x+1)*N + z];

                t.red_next_ptrdiff  = int(red_next - &t);
                t.blue_next_ptrdiff = int(blue_next  - &t);
                t.x = x;
                t.y = -1;
                t.z = z;
                if (Border <= x and x < N - Border
                and Border <= z and z < N - Border) {
                    maybe_fill_tile(t, red_category, blue_category);
                }
            }
        }

        for (int x = 0; x < N; ++x) {
            for (int z = 0; z < N; ++z) {
                tile& t = pos_y_face[x*N + z];

                red_next = z == N-1 ?
                          &pos_z_face[x*N + N-1] : &pos_y_face[x*N + z+1];
                blue_next = x == 0 ?
                          &neg_x_face[(N-1)*N + z] : &pos_y_face[(x-1)*N + z];

                t.red_next_ptrdiff  = int(red_next - &t);
                t.blue_next_ptrdiff = int(blue_next  - &t);
                t.x = x;
                t.y = N;
                t.z = z;
                if (Border <= x and x < N - Border
                and Border <= z and z < N - Border) {
                    maybe_fill_tile(t, red_category, blue_category);
                }
            }
        }

        for (int x = 0; x < N; ++x) {
            for (int y = 0; y < N; ++y) {
                tile& t = neg_z_face[x*N + y];

                red_next = y == N-1 ?
                          &pos_y_face[x*N + 0] : &neg_z_face[x*N + y+1];
                green_next = x == N-1 ?
                          &pos_x_face[y*N + 0] : &neg_z_face[(x+1)*N + y];

                t.red_next_ptrdiff  = int(red_next - &t);
                t.green_next_ptrdiff = int(green_next - &t);
                t.x = x;
                t.y = y;
                t.z = -1;
                if (Border <= x and x < N - Border
                and Border <= y and y < N - Border) {
                    maybe_fill_tile(t, red_category, green_category);
                }
            }
        }

        for (int x = 0; x < N; ++x) {
            for (int y = 0; y < N; ++y) {
                tile& t = pos_z_face[x*N + y];

                red_next = y == 0 ?
                          &neg_y_face[x*N + N-1] : &pos_z_face[x*N + y-1];
                green_next = x == 0 ?
                          &neg_x_face[y*N + N-1] : &pos_z_face[(x-1)*N + y];

                t.red_next_ptrdiff  = int(red_next - &t);
                t.green_next_ptrdiff = int(green_next - &t);
                t.x = x;
                t.y = y;
                t.z = N;
                if (Border <= x and x < N - Border
                and Border <= y and y < N - Border) {
                    maybe_fill_tile(t, red_category, green_category);
                }
            }
        }
    }

    template <uint16_t ColorCategory>
    void update_tile(tile& t) {
        auto relptr = ColorCategory == red_category ? t.red_next_ptrdiff :
                      ColorCategory == green_category ? t.green_next_ptrdiff :
                      ColorCategory == blue_category ? t.blue_next_ptrdiff : 0;
        tile& next = (&t)[relptr];
        assert(&*tile_array.begin() <= &next && &next < &*tile_array.end());
        if (next.old_category == blank && t.old_category == ColorCategory) {
            next.current_category = ColorCategory;
            t.current_category = blank;
            next.color = t.color;
            t.color = blank;
            the_world.set_block(t.x, t.y, t.z, t.color);
            the_world.set_block(next.x, next.y, next.z, next.color);
        }
    }

    void update()
    {
        for (tile& t : tile_array) {
            t.old_category = t.current_category;
        }
        switch (tick_color) {
          default: assert(0);
          break; case red_category:
            tick_color = green_category;
            for (tile& t : tile_array) {
                update_tile<red_category>(t);
            }
          break; case green_category:
            tick_color = blue_category;
            for (tile& t : tile_array) {
                update_tile<green_category>(t);
            }
          break; case blue_category:
            tick_color = red_category;
            for (tile& t : tile_array) {
                update_tile<blue_category>(t);
            }
        }
    }
};

bool do_it;

bool handle_controls(
    glm::vec3* eye_ptr, glm::vec3* forward_normal_vector_ptr,
    glm::mat4* view_ptr, glm::mat4* proj_ptr, float dt)
{
    glm::mat4& view = *view_ptr;
    glm::mat4& projection = *proj_ptr;
    static bool w, a, s, d, down, up, sprint;
    static bool left_mouse, right_mouse;
    static float theta = 1.5707f, phi = 1.5707f;
    static float mouse_x, mouse_y;
    glm::vec3& eye = *eye_ptr;
    static float camera_speed_multiplier = 1.0f;

    static std::unordered_map<SDL_Scancode, SDL_Scancode> scancode_map;
    auto map_scancode = [] (SDL_Scancode in)
    {
        SDL_Scancode out = scancode_map[in];
        if (out == 0) {
            scancode_map[in] = out;
            return in;
        }
        return out;
    };

    bool no_quit = true;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
          default:
          break; case SDL_KEYDOWN:
            switch (map_scancode(event.key.keysym.scancode)) {
              default:
              break; case SDL_SCANCODE_W: w = true;
              break; case SDL_SCANCODE_A: a = true;
              break; case SDL_SCANCODE_S: s = true;
              break; case SDL_SCANCODE_D: d = true;
              break; case SDL_SCANCODE_SPACE: up = true;
              break; case SDL_SCANCODE_LCTRL: down = true;
              break; case SDL_SCANCODE_LSHIFT: sprint = true;
              break; case SDL_SCANCODE_B: chunk_debug = !chunk_debug;
              break; case SDL_SCANCODE_LEFTBRACKET:
                  chunks_to_draw -= (sprint ? 20 : 1);
                  if (chunks_to_draw < 0) chunks_to_draw = -1;
              break; case SDL_SCANCODE_RIGHTBRACKET:
                  chunks_to_draw += (sprint ? 20 : 1);
              break; case SDL_SCANCODE_BACKSPACE:
                  chunks_to_draw = -1;
              break; case SDL_SCANCODE_1: case SDL_SCANCODE_2:
                     case SDL_SCANCODE_3: case SDL_SCANCODE_4:
                     case SDL_SCANCODE_5: case SDL_SCANCODE_6:
                     case SDL_SCANCODE_7: case SDL_SCANCODE_8:
                     case SDL_SCANCODE_9: case SDL_SCANCODE_0:
                camera_speed_multiplier = pow(
                    2, event.key.keysym.scancode - SDL_SCANCODE_3);
              break; case SDL_SCANCODE_Q:
                camera_speed_multiplier *= 0.5f;
              break; case SDL_SCANCODE_E:
                camera_speed_multiplier *= 2.0f;
              break; case SDL_SCANCODE_O:
                eye = glm::vec3(0, 0, 0);
              break; case SDL_SCANCODE_COMMA:
                w = a = s = d = up = down = sprint = false;
                scancode_map[SDL_SCANCODE_U] = SDL_SCANCODE_W;
                scancode_map[SDL_SCANCODE_P] = SDL_SCANCODE_A;
                scancode_map[SDL_SCANCODE_SPACE] = SDL_SCANCODE_S;
                scancode_map[SDL_SCANCODE_A] = SDL_SCANCODE_D;
                scancode_map[SDL_SCANCODE_COMMA] = SDL_SCANCODE_Q;
                scancode_map[SDL_SCANCODE_I] = SDL_SCANCODE_E;
                scancode_map[SDL_SCANCODE_Z] = SDL_SCANCODE_O;
                scancode_map[SDL_SCANCODE_O] = SDL_SCANCODE_LSHIFT;
                scancode_map[SDL_SCANCODE_LALT] = SDL_SCANCODE_LCTRL;
                scancode_map[SDL_SCANCODE_LCTRL] = SDL_SCANCODE_SPACE;
              break; case SDL_SCANCODE_PERIOD:
                scancode_map.clear();
              break; case SDL_SCANCODE_K:
                do_it = true;
              break; case SDL_SCANCODE_BACKSLASH:
                the_world.chunk_map.clear();
              break; case SDL_SCANCODE_MINUS:
                raycast_distance_threshold *= 0.5f;
                printf("raycast_distance_threshold %.1f\n",
                    raycast_distance_threshold);
              break; case SDL_SCANCODE_EQUALS:
                raycast_distance_threshold *= 2;
                printf("raycast_distance_threshold %.1f\n",
                    raycast_distance_threshold);
            }

          break; case SDL_KEYUP:
            switch (map_scancode(event.key.keysym.scancode)) {
              default:
              break; case SDL_SCANCODE_W: w = false;
              break; case SDL_SCANCODE_A: a = false;
              break; case SDL_SCANCODE_S: s = false;
              break; case SDL_SCANCODE_D: d = false;
              break; case SDL_SCANCODE_SPACE: up = false;
              break; case SDL_SCANCODE_LCTRL: down = false;
              break; case SDL_SCANCODE_LSHIFT: sprint = false;
            }
          break; case SDL_MOUSEWHEEL:
            phi -= event.wheel.y * 5e-2f;
            theta -= event.wheel.x * 5e-2f;
          break; case SDL_MOUSEBUTTONDOWN:
            if (event.button.button == SDL_BUTTON_RIGHT) {
                right_mouse = true;
            }
            if (event.button.button == SDL_BUTTON_LEFT) {
                left_mouse = true;
            }
            mouse_x = event.button.x;
            mouse_y = event.button.x;
          break; case SDL_MOUSEBUTTONUP:
            if (event.button.button == SDL_BUTTON_RIGHT) {
                right_mouse = false;
            }
            if (event.button.button == SDL_BUTTON_LEFT) {
                left_mouse = false;
            }
            mouse_x = event.button.x;
            mouse_y = event.button.x;
          break; case SDL_MOUSEMOTION:
            mouse_x = event.motion.x;
            mouse_y = event.motion.y;
            if (right_mouse) {
                theta += event.motion.xrel * 1.5e-3;
                phi += event.motion.yrel * 1.5e-3;
            }
          break; case SDL_WINDOWEVENT:
            if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED ||
                event.window.event == SDL_WINDOWEVENT_RESIZED) {
                    screen_x = event.window.data1;
                    screen_y = event.window.data2;
            }
          break; case SDL_QUIT:
            no_quit = false;
        }
    }

    if (phi < 0.01f) phi = 0.01f;
    if (phi > 3.14f) phi = 3.14f;

    glm::vec3 forward_normal_vector(
        sinf(phi) * cosf(theta),
        cosf(phi),
        sinf(phi) * sinf(theta)
    );
    *forward_normal_vector_ptr = forward_normal_vector;

    // Free-camera mode.
    auto right_vector = glm::cross(forward_normal_vector, glm::vec3(0,1,0));
    right_vector = glm::normalize(right_vector);
    auto up_vector = glm::cross(right_vector, forward_normal_vector);

    float V = dt * camera_speed * camera_speed_multiplier;
    if (sprint) V *= 10;
    eye += V * right_vector * (float)(d - a);
    eye += V * forward_normal_vector * (float)(w - s);
    eye += V * up_vector * (float)(up - down);

    if (left_mouse) {
        theta += dt * 5 * (mouse_x - screen_x*0.5f) / screen_y;
        phi +=   dt * 5 * (mouse_y - screen_y*0.5f) / screen_y;
    }

    view = glm::lookAt(eye, eye+forward_normal_vector, glm::vec3(0,1,0));

    projection = glm::perspective(
        fovy_radians,
        float(screen_x)/screen_y,
        near_plane,
        far_plane
    );

    return no_quit;
}

void update_window_title(glm::vec3 eye)
{
    std::string title = "Voxels (";
    title += std::to_string(int(rintf(eye.x)));
    title += " ";
    title += std::to_string(int(rintf(eye.y)));
    title += " ";
    title += std::to_string(int(rintf(eye.z)));
    title += ") ";
    title += std::to_string(int(rintf(current_fps)));
    title += " FPS ";
    title += std::to_string(chunk::loaded_vbo);
    title += " chunk VBO ";
    SDL_SetWindowTitle(window, title.c_str());
}

void add_random_walks(int walks, int length, std::mt19937& rng)
{
    for (int w = 0; w < walks; ++w) {
        uint16_t blue = rng() >> 28 | 16;
        uint16_t red = rng() >> 28 | 16;
        uint16_t green_base = rng() >> 28;
        int x = 0, y = 0, z = 0;
        for (int i = 0; i < length; ++i) {
            switch (rng() % 6) {
                case 0: x++; break;
                case 1: y++; break;
                case 2: z++; break;
                case 3: x--; break;
                case 4: y--; break;
                case 5: z--; break;
            }
            uint16_t green = (rng() >> 28) + green_base;
            uint16_t color = red << 11 | green << 6 | blue << 1 | 1;
            the_world.set_block(x, y, z, color);
        }
    }
}

int Main(int, char** argv)
{
    argv0 = argv[0];

    window = SDL_CreateWindow(
        "",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        screen_x, screen_y,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
    );
    if (window == nullptr) {
        panic("Could not initialize window", SDL_GetError());
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);

    auto context = SDL_GL_CreateContext(window);
    if (context == nullptr) {
        panic("Could not create OpenGL context", SDL_GetError());
    }
    if (SDL_GL_MakeCurrent(window, context) < 0) {
        panic("SDL OpenGL context error", SDL_GetError());
    }
    if (!gladLoadGL()) {
        panic("gladLoadGL failure", "gladLoadGL failure");
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    std::mt19937 rng;

    auto print_world_size = [&] ()
    {
        double block_count = 0;
        double chunk_count = 0;
        for (auto& a_chunk : the_world.chunk_map) {
            block_count += a_chunk.second->get_opaque_block_count();
            ++chunk_count;
        }
        printf("%.0f blocks   %.0f chunks\n", block_count, chunk_count);
    };

    bool no_quit = true;

    glm::vec3 eye(0, 0, 0), forward_normal_vector;
    glm::mat4 view_matrix, proj_matrix;

    auto previous_update = SDL_GetTicks();
    auto previous_congestion_update = SDL_GetTicks();
    auto previous_fps_print = SDL_GetTicks();
    auto previous_handle_controls = SDL_GetTicks();
    int frames = 0;

    static congestion_model<365, 8> the_congestion_model(rng, 0.8);

    while (no_quit) {
        update_window_title(eye);
        auto current_tick = SDL_GetTicks();
        if (current_tick >= previous_update + 16) {
            float dt = 0.001f * (current_tick - previous_handle_controls);
            dt = std::min(dt, 1/20.0f);
            no_quit = handle_controls(
                &eye, &forward_normal_vector, &view_matrix, &proj_matrix, dt);
            previous_handle_controls = current_tick;
            previous_update += 16;
            if (current_tick - previous_congestion_update > 75) {
                the_congestion_model.update();
                previous_congestion_update += 75;
            }
            if (current_tick - previous_update > 100) {
                previous_update = current_tick;
                previous_congestion_update = current_tick;
            }

            ++frames;
            if (current_tick >= previous_fps_print + 250) {
                current_fps = 1000.0 * frames
                            / (current_tick-previous_fps_print);
                previous_fps_print = current_tick;
                frames = 0;
            }
        }
        if (do_it) {
            for (int i = 0; i < 1000; ++i) the_congestion_model.update();
            do_it = false;
        }
        // if (do_it) {
        //     add_random_walks(1, 100000, rng);
        //     do_it = false;
        //     print_world_size();
        // }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, screen_x, screen_y);
        draw_scene(eye, forward_normal_vector, view_matrix, proj_matrix);
        SDL_GL_SwapWindow(window);
        PANIC_IF_GL_ERROR;
    }

    return 0;
}

} // end anonymous namespace

int main(int argc, char** argv)
{
    return Main(argc, argv);
}
