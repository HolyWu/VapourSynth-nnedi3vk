project('volk', 'c',
    default_options: ['default_library=static', 'warning_level=0', 'c_std=c11'],
)

volk = library('volk',
    files('volk.c'),
    dependencies: dependency('vulkan-headers'),
)

volk_dep = declare_dependency(
    include_directories: include_directories('.', is_system: true),
    link_with: volk,
)

meson.override_dependency('vulkan-volk', volk_dep)
