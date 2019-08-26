param (
    [string]$filename = $(throw "-filename is required.")
)

$ErrorActionPreference="Stop"

# Get-FileHash is sha256 by default, but explicit is better!
# Most (all?) of powerhshell's string output stuff is the wildly terrible
# UTf-16, which unnecessarily inflates the output and makes it more annoying
# to read, so we force it to ASCII, and tell it remove newlines so the actual
# contents are exactly 64 bytes, like god intended
(Get-FileHash "${filename}" -Algorithm SHA256).Hash | Out-File -Encoding ASCII -NoNewline "${filename}.sha256"