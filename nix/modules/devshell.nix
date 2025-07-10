{ inputs, ... }: {
  perSystem = { config, self', pkgs, system, lib, ... }:
    let
      npkgs = import inputs.nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

    in
    {
      devShells.default = pkgs.mkShell {
        name = "whysper-shell";
        buildInputs = with npkgs; [
          alsa-lib

          pkg-config # Required for linking
          cudatoolkit
          cudaPackages.cuda_cudart # Critical runtime library
          # linuxPackages.nvidia_x11
        ];
        shellHook = ''
                export LD_LIBRARY_PATH=${npkgs.linuxPackages.nvidia_x11}/lib:${npkgs.cudaPackages.cudatoolkit}/lib:${npkgs.cudaPackages.cuda_cudart}/lib:$LD_LIBRARY_PATH
          export CUDA_PATH=${npkgs.cudaPackages.cudatoolkit}
          export PKG_CONFIG_PATH=${npkgs.cudaPackages.cudatoolkit}/lib/pkgconfig:$PKG_CONFIG_PATH
                # export LD_LIBRARY_PATH=${npkgs.cudatoolkit}/lib$LD_LIBRARY_PATH
                # export CUDA_PATH=${npkgs.cudaPackages.cudatoolkit}
        '';
        inputsFrom = [
          self'.devShells.rust
          config.pre-commit.devShell # See ./nix/modules/pre-commit.nix
        ];
        packages = with pkgs; [
          just
          nixd # Nix language server
          bacon
          wget
          ffmpeg
          # config.process-compose.cargo-doc-live.outputs.package
        ];
      };
    };
}
