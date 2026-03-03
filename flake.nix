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

        # Set LD_LIBRARY_PATH so sounddevice can find PortAudio at runtime
        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.portaudio ]}:$LD_LIBRARY_PATH"
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
