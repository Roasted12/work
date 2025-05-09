{pkgs}: {
  deps = [
    pkgs.rustc
    pkgs.cargo
    pkgs.spdlog
    pkgs.nlohmann_json
    pkgs.muparserx
    pkgs.fmt
    pkgs.catch2
    pkgs.tk
    pkgs.tcl
    pkgs.qhull
    pkgs.pkg-config
    pkgs.gtk3
    pkgs.gobject-introspection
    pkgs.ghostscript
    pkgs.freetype
    pkgs.ffmpeg-full
    pkgs.cairo
    pkgs.glibcLocales
    pkgs.postgresql
    pkgs.openssl
  ];
}
