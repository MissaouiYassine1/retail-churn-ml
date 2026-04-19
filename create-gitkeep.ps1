# Windows
Get-ChildItem -Recurse -Directory | Where-Object {
    $_.FullName -notmatch "\\.git|venv|__pycache__" -and
    (Get-ChildItem $_.FullName -Force | Measure-Object).Count -eq 0
} | ForEach-Object {
    New-Item -Path "$($_.FullName)\.gitkeep" -ItemType File -Force
}