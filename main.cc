#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include <utility>
using std::swap;

#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "gl_core_3_3.h"
#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

constexpr int chunk_size = 32;
#define CHUNK_SIZE_STR "32"

constexpr int chunk_view_radius = 7;

namespace {

constexpr float
    fovy_radians = 1.0f,
    near_plane = 0.3f,
    far_plane = 700.0f,
    camera_speed = 8e-2;

int screen_x = 1280, screen_y = 960;
SDL_Window* window = nullptr;
std::string argv0;

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

static const float chunk_vertices[32] =
{
    0, chunk_size, chunk_size, 1,
    0, 0, chunk_size, 1,
    chunk_size, 0, chunk_size, 1,
    chunk_size, chunk_size, chunk_size, 1,
    0, chunk_size, 0, 1,
    0, 0, 0, 1,
    chunk_size, 0, 0, 1,
    chunk_size, chunk_size, 0, 1,
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
"out vec4 model_space_position;\n"
"uniform vec3 chunk_offset;\n"
"uniform mat4 view_matrix;\n"
"uniform mat4 proj_matrix;\n"
"void main() {\n"
    "gl_Position = proj_matrix * view_matrix * \n"
        "(in_vertex + vec4(chunk_offset, 0));\n"
    "model_space_position = in_vertex;\n"
"}\n";

static const char chunk_fs_source[] =
"#version 330\n"
"in vec4 model_space_position;\n"
"uniform vec3 eye_in_model_space;\n"
"uniform sampler3D chunk_blocks;\n"
"uniform vec3 chunk_offset;\n"
"uniform bool chunk_debug;\n"
"uniform mat4 view_matrix;\n"
"uniform mat4 proj_matrix;\n"
"out vec4 color;\n"
"void main() {\n"
"if (!chunk_debug) {\n"
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
    "int x_init = xm > 0 ? 0 : " CHUNK_SIZE_STR ";\n"
    "int x_end = xm > 0 ? " CHUNK_SIZE_STR " : 0;\n"
    "int x_step = xm > 0 ? 1 : -1;\n"
    "float x_fudge = xm > 0 ? .25 : -.25;\n"
    "for (int x = x_init; x != x_end; x += x_step) {\n"
        "float t = (x - x0) / xm;\n"
        "float y = y0 + ym * t;\n"
        "float z = z0 + zm * t;\n"
        "if (y < 0 || y > " CHUNK_SIZE_STR ") continue;\n"
        "if (z < 0 || z > " CHUNK_SIZE_STR ") continue;\n"
        "vec3 texcoord = vec3(x + x_fudge, y, z) * rcp;\n"
        "vec4 lookup_color = texture(chunk_blocks, texcoord);\n"
        "if (lookup_color.a > 0 && t > 0) {\n"
            "if (best_t > t) {\n"
                "best_t = t;\n"
                "best_color = lookup_color;\n"
                "best_coord = vec3(x,y,z);\n"
                "if (y - floor(y + .1) < .1 || z - floor(z + .1) < .1) {\n"
                    "best_color.rgb *= 0.5;\n"
                "}\n"
            "}\n"
            "break;\n"
        "}\n"
    "}\n"
    "int y_init = ym > 0 ? 0 : " CHUNK_SIZE_STR ";\n"
    "int y_end = ym > 0 ? " CHUNK_SIZE_STR " : 0;\n"
    "int y_step = ym > 0 ? 1 : -1;\n"
    "float y_fudge = ym > 0 ? .05 : -.05;\n"
    "for (int y = y_init; y != y_end; y += y_step) {\n"
        "float t = (y - y0) / ym;\n"
        "float x = x0 + xm * t;\n"
        "float z = z0 + zm * t;\n"
        "if (x < 0 || x > " CHUNK_SIZE_STR ") continue;\n"
        "if (z < 0 || z > " CHUNK_SIZE_STR ") continue;\n"
        "vec3 texcoord = vec3(x, y + y_fudge, z) * rcp;\n"
        "vec4 lookup_color = texture(chunk_blocks, texcoord);\n"
        "if (lookup_color.a > 0 && t > 0) {\n"
            "if (best_t > t) {\n"
                "best_t = t;\n"
                "best_color = lookup_color;\n"
                "best_coord = vec3(x,y,z);\n"
                "if (x - floor(x + .1) < .1 || z - floor(z + .1) < .1) {\n"
                    "best_color.rgb *= 0.5;\n"
                "}\n"
            "}\n"
            "break;\n"
        "}\n"
    "}\n"
    "int z_init = zm > 0 ? 0 : " CHUNK_SIZE_STR ";\n"
    "int z_end = zm > 0 ? " CHUNK_SIZE_STR " : 0;\n"
    "int z_step = zm > 0 ? 1 : -1;\n"
    "float z_fudge = zm > 0 ? .05 : -.05;\n"
    "for (int z = z_init; z != z_end; z += z_step) {\n"
        "float t = (z - z0) / zm;\n"
        "float x = x0 + xm * t;\n"
        "float y = y0 + ym * t;\n"
        "if (x < 0 || x > " CHUNK_SIZE_STR ") continue;\n"
        "if (y < 0 || y > " CHUNK_SIZE_STR ") continue;\n"
        "vec3 texcoord = vec3(x, y, z + z_fudge) * rcp;\n"
        "vec4 lookup_color = texture(chunk_blocks, texcoord);\n"
        "if (lookup_color.a > 0 && t > 0) {\n"
            "if (best_t > t) {\n"
                "best_t = t;\n"
                "best_color = lookup_color;\n"
                "best_coord = vec3(x,y,z);\n"
                "if (x - floor(x + .1) < .1 || y - floor(y + .1) < .1) {\n"
                    "best_color.rgb *= 0.5;\n"
                "}\n"
            "}\n"
            "break;\n"
        "}\n"
    "}\n"
    "if (best_color.a == 0) discard;\n"
    "color = best_color;\n"
    //"vec4 v = proj_matrix * view_matrix * vec4(best_coord + chunk_offset, 1);\n"
    //"gl_FragDepth = .3;\n"
"} else {\n"
    // "color = texture(chunk_blocks, model_space_position.xyz / 16.0);\n"
    "int x_floor = int(floor(model_space_position.x));\n"
    "int y_floor = int(floor(model_space_position.y));\n"
    "int z_floor = int(floor(model_space_position.z));\n"
    "int n = (x_floor + y_floor + z_floor) % 8;\n"
    "color = vec4((n & 1) != 0 ? 1 : 0, (n & 2) != 0 ? 1 : 0, (n & 4) != 0 ? 1 : 0, 1);\n"
"}\n"
"}\n";

class chunk
{
    bool dirty = false;
    uint16_t blocks[chunk_size][chunk_size][chunk_size] = {};
    GLuint texture_name = 0;
    int32_t opaque_block_count = 0;
  public:
    const glm::vec3 position;

    chunk(glm::vec3 in_position) : position(in_position)
    {

    }

    GLuint get_texture_name()
    {
        if (texture_name == 0)
        {
            dirty = true;
            glGenTextures(1, &texture_name);
            glBindTexture(GL_TEXTURE_3D, texture_name);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            PANIC_IF_GL_ERROR;
        }
        if (dirty)
        {
            glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA,
                chunk_size, chunk_size, chunk_size, 0,
                GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, blocks);
            dirty = false;
            PANIC_IF_GL_ERROR;
        }
        return texture_name;
    }

    void unload_texture()
    {
        if (texture_name != 0) {
            glDeleteTextures(1, &texture_name);
            texture_name = 0;
            PANIC_IF_GL_ERROR;
        }
    }

    int32_t get_opaque_block_count() const
    {
        return opaque_block_count;
    }

    void set_block(size_t x, size_t y, size_t z, uint16_t block)
    {
        assert(x < chunk_size);
        assert(y < chunk_size);
        assert(z < chunk_size);
        const int old_alpha = int(blocks[x][y][z] & 1);
        const int new_alpha = int(block & 1);
        blocks[x][y][z] = block;
        opaque_block_count += (new_alpha - old_alpha);
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
    }

    chunk(chunk&&) = delete;
};

bool chunk_debug = false;

void draw_chunk(
    chunk& c, glm::vec3 eye, glm::mat4 view_matrix, glm::mat4 proj_matrix)
{
    if (c.get_opaque_block_count() == 0) {
        return;
    }

    static GLuint vao = 0;
    static GLuint program_id;
    static GLuint vertex_buffer_id;
    static GLuint element_buffer_id;
    static GLint chunk_offset_id;
    static GLint view_matrix_id;
    static GLint proj_matrix_id;
    static GLint chunk_blocks_id;
    static GLint eye_in_model_space_id;
    static GLint chunk_debug_id;

    if (vao == 0) {
        program_id = make_program(chunk_vs_source, chunk_fs_source);
        chunk_offset_id = glGetUniformLocation(program_id, "chunk_offset");
        view_matrix_id = glGetUniformLocation(program_id, "view_matrix");
        proj_matrix_id = glGetUniformLocation(program_id, "proj_matrix");
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
    // printf("(%f %f %f)\n", eye_in_model_space.x, eye_in_model_space.y, eye_in_model_space.z);

    glUseProgram(program_id);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, c.get_texture_name());

    glUniform1i(chunk_blocks_id, 0);
    glUniform3fv(chunk_offset_id, 1, &c.position[0]);
    glUniformMatrix4fv(view_matrix_id, 1, 0, &view_matrix[0][0]);
    glUniformMatrix4fv(proj_matrix_id, 1, 0, &proj_matrix[0][0]);
    glUniform3fv(eye_in_model_space_id, 1, &eye_in_model_space[0]);
    glUniform1i(chunk_debug_id, chunk_debug);

    glBindVertexArray(vao);
    //glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (void*)0);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, (void*)0);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_3D, 0);

    PANIC_IF_GL_ERROR;
}

class world
{
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
            return chunk_map[key].get();
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
        int x_chunk = x - int(x_in_chunk);
        int y_chunk = y - int(y_in_chunk);
        int z_chunk = z - int(z_in_chunk);
        chunk_ref(x_chunk, y_chunk, z_chunk).set_block(
            x_in_chunk, y_in_chunk, z_in_chunk, block);
    }
};

std::vector<std::unique_ptr<chunk>> chunk_vector;

std::unique_ptr<chunk> random_chunk(
    glm::vec3 chunk_position, std::mt19937& rng)
{
    auto color = (rng() >> 16) | 1;
    auto chunk_ptr = std::make_unique<chunk>(chunk_position);
    for (int i = 0; i < 512; ++i) {
        auto x = rng() >> 27;
        auto y = rng() >> 27;
        auto z = rng() >> 27;
        chunk_ptr->set_block(x, y, z, color);
    }
    for (int i = 0; i < 2048; ++i) {
        color = (rng() >> 16) | 1;
        auto x = rng() >> 27;
        auto y = rng() >> 27;
        auto z = rng() >> 27;
        chunk_ptr->set_block(x, y, z, color);
    }
    return chunk_ptr;
}

void draw_scene(
    glm::vec3 eye,
    glm::vec3 forward_normal_vector,
    glm::mat4 view_matrix,
    glm::mat4 proj_matrix)
{
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    draw_skybox(view_matrix, proj_matrix);
    chunk* chunk_eye_is_inside = nullptr;
    glm::vec3 fudged_eye = eye + forward_normal_vector * 0.5f;
    // Todo: sort from nearest to furthest (reverse painters)
    for (auto& chunk_ptr : chunk_vector) {
        if (chunk_ptr->in_chunk(fudged_eye)) {
            chunk_eye_is_inside = chunk_ptr.get();
        }
        else {
            draw_chunk(*chunk_ptr, eye, view_matrix, proj_matrix);
        }
    }
    if (chunk_eye_is_inside != nullptr) {
        glDisable(GL_DEPTH_TEST);
        draw_chunk(*chunk_eye_is_inside, eye, view_matrix, proj_matrix);
        glEnable(GL_DEPTH_TEST);
    }
}

bool handle_controls(
    glm::vec3* eye_ptr, glm::vec3* forward_normal_vector_ptr,
    glm::mat4* view_ptr, glm::mat4* proj_ptr)
{
    glm::mat4& view = *view_ptr;
    glm::mat4& projection = *proj_ptr;
    static bool w, a, s, d, q, e, space;
    static bool right_mouse;
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
              break; case SDL_SCANCODE_Q: q = true;
              break; case SDL_SCANCODE_E: e = true;
              break; case SDL_SCANCODE_SPACE:  space = true;
              break; case SDL_SCANCODE_B: chunk_debug = !chunk_debug;
              break; case SDL_SCANCODE_1: case SDL_SCANCODE_2:
                     case SDL_SCANCODE_3: case SDL_SCANCODE_4:
                     case SDL_SCANCODE_5: case SDL_SCANCODE_6:
                     case SDL_SCANCODE_7: case SDL_SCANCODE_8:
                     case SDL_SCANCODE_9: case SDL_SCANCODE_0:
                camera_speed_multiplier = pow(
                    2, event.key.keysym.scancode - SDL_SCANCODE_5);
              break; case SDL_SCANCODE_COMMA:
                w = a = s = d = q = e = space = false;
                scancode_map[SDL_SCANCODE_O] = SDL_SCANCODE_Q;
                scancode_map[SDL_SCANCODE_U] = SDL_SCANCODE_W;
                scancode_map[SDL_SCANCODE_P] = SDL_SCANCODE_E;
                scancode_map[SDL_SCANCODE_E] = SDL_SCANCODE_A;
                scancode_map[SDL_SCANCODE_SPACE] = SDL_SCANCODE_S;
                scancode_map[SDL_SCANCODE_A] = SDL_SCANCODE_D;
                scancode_map[SDL_SCANCODE_LCTRL] = SDL_SCANCODE_SPACE;
            }

          break; case SDL_KEYUP:
            switch (map_scancode(event.key.keysym.scancode)) {
              default:
              break; case SDL_SCANCODE_W: w = false;
              break; case SDL_SCANCODE_A: a = false;
              break; case SDL_SCANCODE_S: s = false;
              break; case SDL_SCANCODE_D: d = false;
              break; case SDL_SCANCODE_Q: q = false;
              break; case SDL_SCANCODE_E: e = false;
              break; case SDL_SCANCODE_SPACE:  space = false;
            }
          break; case SDL_MOUSEWHEEL:
            phi -= event.wheel.y * 5e-2f;
            theta -= event.wheel.x * 5e-2f;
          break; case SDL_MOUSEBUTTONDOWN:
            if (event.button.button == SDL_BUTTON_RIGHT) {
                right_mouse = true;
            }
            mouse_x = event.button.x;
            mouse_y = event.button.x;
          break; case SDL_MOUSEBUTTONUP:
            if (event.button.button == SDL_BUTTON_RIGHT) {
                right_mouse = false;
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

    float V = camera_speed * camera_speed_multiplier;
    eye += V * right_vector * (float)(d - a);
    eye += V * forward_normal_vector * (float)(w - s);
    eye += V * up_vector * (float)(e - q);

    if (space) {
        theta += 8e-2 * (mouse_x - screen_x*0.5f) / screen_y;
        phi +=   8e-2 * (mouse_y - screen_y*0.5f) / screen_y;
    }

    view = glm::lookAt(eye, eye+forward_normal_vector, glm::vec3(0,1,0));

    projection = glm::perspective(
        fovy_radians,
        float(screen_x)/screen_y,
        near_plane,
        far_plane
    );

    float y_plane_radius = tanf(fovy_radians / 2.0f);
    float x_plane_radius = y_plane_radius * screen_x / screen_y;
    float mouse_vcs_x = x_plane_radius * (2.0f * mouse_x / screen_x - 1.0f);
    float mouse_vcs_y = y_plane_radius * (1.0f - 2.0f * mouse_y / screen_y);
    glm::vec4 mouse_vcs(mouse_vcs_x, mouse_vcs_y, -1.0f, 1.0f);
//    glm::vec4 mouse_wcs = glm::inverse(view) * mouse_vcs;

    return no_quit;
}

int Main(int, char** argv)
{
    argv0 = argv[0];

    window = SDL_CreateWindow(
        "Bouncy",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        screen_x, screen_y,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
    );
    if (window == nullptr) {
        panic("Could not initialize window", SDL_GetError());
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

    auto context = SDL_GL_CreateContext(window);
    if (context == nullptr) {
        panic("Could not create OpenGL context", SDL_GetError());
    }
    if (SDL_GL_MakeCurrent(window, context) < 0) {
        panic("SDL OpenGL context error", SDL_GetError());
    }

    ogl_LoadFunctions();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glEnable(GL_DEPTH_TEST);
    // glEnable(GL_CULL_FACE);
    // glCullFace(GL_BACK);

    std::mt19937 rng;
    const float bounds = chunk_view_radius * chunk_size;
    double block_count = 0;
    for (float x = 0; x < bounds; x += chunk_size) {
        for (float y = 0; y < bounds; y += chunk_size) {
            for (float z = 0; z < bounds; z += chunk_size) {
                auto unique_chunk = random_chunk(glm::vec3(x, y, z), rng);
                block_count += unique_chunk->get_opaque_block_count();
                chunk_vector.push_back(std::move(unique_chunk));
            }
        }
    }
    printf("%.0f blocks\n", block_count);

    bool no_quit = true;

    glm::vec3 eye(0, 0, 0), forward_normal_vector;
    glm::mat4 view_matrix, proj_matrix;

    auto previous_update = SDL_GetTicks();
    auto previous_fps_print = SDL_GetTicks();
    int frames = 0;

    while (no_quit) {
        auto current_tick = SDL_GetTicks();
        if (current_tick >= previous_update + 16) {
            no_quit = handle_controls(
                &eye, &forward_normal_vector, &view_matrix, &proj_matrix);
            previous_update += 16;
            if (current_tick - previous_update > 100) {
                previous_update = current_tick;
            }

            ++frames;
            if (current_tick >= previous_fps_print + 2000) {
                float fps = 1000.0 * frames / (current_tick-previous_fps_print);
                printf("%4.1f FPS\n", fps);
                previous_fps_print = current_tick;
                frames = 0;
            }
        }
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
