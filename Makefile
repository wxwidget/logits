# Historical executables are opt-in; modern core is the default.
CMAKE ?= cmake
BUILD_DIR ?= build/release
.PHONY: all legacy test clean
all:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build $(BUILD_DIR) --parallel 2
legacy:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DLOGITLAB_LEGACY=ON
	$(CMAKE) --build $(BUILD_DIR) --parallel 2
test: all
	ctest --test-dir $(BUILD_DIR) --output-on-failure
clean:
	$(CMAKE) --build $(BUILD_DIR) --target clean
