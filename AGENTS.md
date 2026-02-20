# `AGENTS.md`

## Project Structure

- Any code that interacts with the Slurm REST API goes in the `slurmrestapi` directory.
- The `slurmrestapi` module returns a `pydantic` model representing the data fetched 
  from the API but does not perform any data processing.
- Bootstrapping and dependency injection happens in the `main.py` file.
- This project uses `uv` as a package manager.
- The exporter is containarized. The `Containerfile` defines how to build the 
  container image.
- The `README.md` file has a Repo Structure section that explains the architecture of 
  the exporter adn the components in more detail.

## Project Guidelines

- The project follows the conventional commits specification. Format the commit 
  messages accordingly.

## Slurm Rest API

Slurm REST API documentation is available at https://slurm.schedmd.com/rest_api.html.

## Running the Code Locally

You can run the exporter locally.

```bash
SLURM_EXPORTER_CONFIG_PATH=config.json uv run uvicorn slurm_prometheus_exporter.server:create_app --factory --host 0.0.0.0 --port 9092
```

You can query metrics using the command below.

```bash
curl http://localhost:9092/metrics
```

Assume that a `config.json` file is present in the project root. If there isn't one, 
ask the user to create one.

## Do

- Use type annotations throughout the codebase.
- Use `ty` to for type checking.
- Use `uv` to manage dependencies.
- Use `ruff` for linting.
- Prefer importing whole modules instead of individual classes or functions, e.g. 
  prefer `from starlette import responses` over 
  `from starlette.responses import JSONResponse`.
- Run the code often, especially after making larger changes.

## Don't

- Avoid importing single classes or functions from modules, prefer importing the 
  entire module.

## Commands

- Check typing using `ty`
    
    ```bash
    ty check
    ```

- Lint code using `ruff`
    
    ```bash
    ruff check
    ```
