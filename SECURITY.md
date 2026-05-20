# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities **privately** by emailing
**openlithohub@163.com**.

Do not open public GitHub issues, discussions, or pull requests for
security matters until a fix has been coordinated.

When reporting, please include:

- A clear description of the vulnerability
- Steps to reproduce or a proof-of-concept
- The affected version(s) or commit hash
- Your assessment of impact (data exposure, code execution, etc.)
- Whether you intend to publicly disclose, and your preferred timeline

## Supported Versions

Only the latest minor release on the `main` branch is actively supported
with security fixes. Older releases may receive fixes at the maintainers'
discretion.

## Disclosure Timeline

| Stage | Target |
|-------|--------|
| Acknowledgement of report | within 7 days |
| Initial assessment | within 14 days |
| Coordinated public disclosure | typically within 90 days of initial report |

We follow a coordinated-disclosure model. We will work with you on
timing and credit for the disclosure, and we ask that you do not
publicly disclose the issue before a fix is available.

## Scope

This policy covers the OpenLithoHub source code in this repository.

Out of scope:

- Vulnerabilities in third-party dependencies (please report those
  upstream — see [NOTICE](NOTICE) for the dependency list)
- Vulnerabilities in datasets accessed via OpenLithoHub adapters
  (those datasets retain their own maintainers — see
  [DATA-LICENSES.md](DATA-LICENSES.md))

## Safe Harbor

Good-faith security research conducted in accordance with this policy
is welcomed. We will not pursue legal action against researchers who:

- Make a good-faith effort to avoid privacy violations and disruption
- Only interact with accounts and data they own or have explicit
  permission to access
- Report vulnerabilities promptly and refrain from public disclosure
  until a fix is coordinated
