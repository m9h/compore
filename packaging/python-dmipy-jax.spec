%global pypi_name compore

Name:           python-compore
Version:        0.1.0
Release:        1%{?dist}
Summary:        JAX-accelerated Microstructure Imaging (Drop-in replacement for dmipy)

# License found in repository root is MIT.
License:        MIT
URL:            https://github.com/AthenaEPI/dmipy
Source0:        %{pypi_name}-%{version}.tar.gz

BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  python3-pytest

%description
JAX-accelerated Microstructure Imaging (Drop-in replacement for dmipy).

%package -n python3-compore
Summary:        %{summary}
Requires:       python3-jax
Requires:       python3-dmipy
Requires:       python3-numpy
# Note: jaxlib GPU support is likely desired but we depend on python3-jax which pulls
# in a CPU-only jaxlib by default on Fedora. Users need to install proper jaxlib for GPU.

%description -n python3-compore
%{description}

%prep
%autosetup -p1 -n compore-%{version}

%generate_buildrequires
%pyproject_buildrequires

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files compore

%check
# Run pytest, excluding Heavyweight benchmarks
%pytest -k 'not Heavyweight'

%files -n python3-compore -f %{pyproject_files}
%doc README.md
%license LICENSE

%changelog
* Tue Jan 13 2026 Antigravity <antigravity@example.com> - 0.1.0-1
- Initial package
