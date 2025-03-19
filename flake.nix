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
                    buildPythonPackage rec {
                      pname = "wrapt_timeout_decorator";
                      version = "1.5.1";
                      
                      src = fetchPypi {
                        inherit pname version;
                        hash = "sha256-CP3V73yWSArRHBLUct4hrNMjWZlvaaUlkpm1QP66RWA=";
                      };
                      
                      # do not run tests
                      doCheck = false;
                      
                      # specific to buildPythonPackage, see its reference
                      pyproject = true;
                      build-system = [
                        setuptools
                        wheel
                      ];
                    }

									 ]))
          ];
        };
      };
    };
}
