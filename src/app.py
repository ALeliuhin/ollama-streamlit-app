if __name__ == "__main__":
    import subprocess
    import sys

    import streamlit.runtime as st_runtime

    try:
        from src.ui import main
    except ImportError:
        from ui import main

    if not st_runtime.exists():
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]]))
    main()
