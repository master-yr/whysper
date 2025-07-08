{ inputs, ... }: {
  perSystem = { config, self', pkgs, lib, ... }: {
    devShells.default = pkgs.mkShell {
      name = "whysper-shell";
      buildInputs = with pkgs; [ alsa-lib ];
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
