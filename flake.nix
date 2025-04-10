{
  description = "DiffSym";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = inputs@{ flake-parts, nixpkgs, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      # This is the list of architectures that work with this project
      imports = [];
      systems = [
        "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin"
      ];
      perSystem = { config, self', inputs', pkgs, system, ... }:
			let
          pkgsWithUnfree = pkgs // {
            config.allowUnfree = true;
          };

			in {
        _module.args.pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        devShells.default = pkgs.mkShell { 
          packages = with pkgsWithUnfree; [
									 (python3.withPackages (pypkgs: with pypkgs; [
										torch-bin
									 	numpy
									 	scipy
									 	matplotlib
									 	tqdm
										jupyterlab
      							python-lsp-server
      							python-lsp-ruff
      							pylsp-mypy
									 ]))
          ];
        };
      };
    };
}
