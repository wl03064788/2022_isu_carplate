# ISU 2020 Spring Graduate Project

![CI](https://github.com/<your-org>/<your-repo>/actions/workflows/ci.yml/badge.svg)
![downloads](https://img.shields.io/github/downloads/atom/atom/total.svg)
![chat](https://img.shields.io/discord/:serverId.svg)

## CI/CD Pipeline

```mermaid
flowchart LR
    A[Push / Pull Request] --> B[GitHub Actions]
    B --> C[Setup Python 3.11]
    C --> D[Validate JSON]
    D --> E[Python Syntax Check]
    E --> F[Build Success]
```

## Project Timeline

```mermaid
gantt
    title A Gantt Diagram

    section Section
    A task           :a1, 2014-01-01, 30d
    Another task     :after a1, 20d
    section Another
    Task in sec      :2014-01-12, 12d
    Another task     :24d
```

> Read more about Mermaid: http://mermaid-js.github.io/mermaid/

##### Write by HsuKC on 23/05/20
###### Tags: `AIoT` `Cloud computing`
