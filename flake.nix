{
  description = "Vision - Image processing pipelines with Google Images and KGIZ";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # Load the uv workspace from uv.lock
        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        # Create package overlay from workspace
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # Python package set with our overlay
        python = pkgs.python312;

        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
            (
              pkgs.lib.composeManyExtensions [
                pyproject-build-systems.overlays.default
                overlay
              ]
            );

        # The vision package (production)
        visionPackage = pythonSet.mkVirtualEnv "vision-env" workspace.deps.default;

        # Development environment with all dev dependencies
        devEnv = pythonSet.mkVirtualEnv "vision-dev-env" workspace.deps.all;
      in
      {
        packages = {
          default = visionPackage;
          vision = visionPackage;
        };

        apps = {
          default = {
            type = "app";
            program = "${visionPackage}/bin/vision";
          };
          vision = {
            type = "app";
            program = "${visionPackage}/bin/vision";
          };
        };

        devShells.default = pkgs.mkShell {
          packages = [
            # Python environment with vision + dev dependencies
            devEnv

            # Package management
            pkgs.uv

            # Linting & formatting
            pkgs.ruff
            pkgs.pyright

            # Testing
            pkgs.python312Packages.pytest
            pkgs.python312Packages.pytest-asyncio

            # Utilities
            pkgs.jq
            pkgs.yq-go
          ];

          shellHook = ''
            # Colors
            CYAN='\033[0;36m'
            GREEN='\033[0;32m'
            YELLOW='\033[0;33m'
            DIM='\033[2m'
            BOLD='\033[1m'
            NC='\033[0m' # No Color

            echo ""
            echo -e "''${CYAN}''${BOLD}  Vision Development Environment''${NC}"
            echo -e "''${DIM}  Image processing pipelines with Google Images and KGIZ''${NC}"
            echo ""
            echo -e "''${GREEN}Python:''${NC}  $(python --version 2>&1 | cut -d' ' -f2)"
            echo -e "''${GREEN}uv:''${NC}      $(uv --version 2>&1 | cut -d' ' -f2)"
            echo -e "''${GREEN}ruff:''${NC}    $(ruff --version 2>&1 | cut -d' ' -f2)"
            echo ""
            echo -e "''${YELLOW}Commands:''${NC}"
            echo -e "  ''${BOLD}vision''${NC}            Run the CLI (try: vision --help)"
            echo -e "  ''${BOLD}uv sync''${NC}           Install/update dependencies"
            echo -e "  ''${BOLD}uv run pytest''${NC}     Run tests"
            echo -e "  ''${BOLD}ruff check .''${NC}      Lint code"
            echo -e "  ''${BOLD}ruff format .''${NC}     Format code"
            echo -e "  ''${BOLD}pyright''${NC}           Type check"
            echo ""

            # Ensure we're using the right Python
            export PYTHONDONTWRITEBYTECODE=1

            # Project root for imports
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
          '';

          env = {
            # Prevent uv from downloading Python
            UV_PYTHON_DOWNLOADS = "never";

            # Use system Python
            UV_PYTHON = "${python}/bin/python";

            # Colored output
            FORCE_COLOR = "1";
          };
        };
      }
    );
}
