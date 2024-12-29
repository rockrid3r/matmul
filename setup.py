from distutils.core import setup, Extension

def main():
    setup(name="matmul",
          version="1.0.0",
          description="Matmul implementation",
          author="rockrid3r",
          author_email="rockrid3r@outlook.com",
          ext_modules=[Extension(
            "matmul", 
            ["matmulmodule.c"],
            language="c",
          )],
    )

if __name__ == "__main__":
    main()
