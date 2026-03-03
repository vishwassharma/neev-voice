{
  description = "Neev Voice - Python CLI voice agent for Hindi-English mixed speech";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        commonPackages = with pkgs; [
          uv
          portaudio
        ];

        # Set LD_LIBRARY_PATH so native Python wheels can find shared libs:
        #   stdenv.cc.cc.lib → libstdc++.so.6 (numpy, scipy C extensions)
        #   portaudio        → libportaudio.so (sounddevice)
        #   zlib             → libz.so (various Python packages)
        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.portaudio
            pkgs.zlib
          ]}:$LD_LIBRARY_PATH"
        '';
      in
      {
        devShells = {
          default = pkgs.mkShell {
            buildInputs = commonPackages ++ [ pkgs.python312 ];
            inherit shellHook;
          };

          ci = pkgs.mkShell {
            buildInputs = commonPackages ++ [
              pkgs.python312
              pkgs.python313
            ];
            inherit shellHook;
          };
        };
      }
    );
}
