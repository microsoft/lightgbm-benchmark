# escape=`

# Use the latest Windows Server Core image with .NET Framework 4.8.
#FROM mcr.microsoft.com/dotnet/framework/sdk:4.8-windowsservercore-ltsc2019
FROM mcr.microsoft.com/azureml/windows-servercore-1809:latest

# Restore the default Windows shell for correct batch processing.
SHELL ["cmd", "/S", "/C"]

RUN `
    # Download the Build Tools bootstrapper.
    curl -SL --output vs_buildtools.exe https://aka.ms/vs/16/release/vs_buildtools.exe `
    `
    # Install Build Tools with the Microsoft.VisualStudio.Workload.AzureBuildTools workload, excluding workloads and components with known issues.
    && (start /w vs_buildtools.exe --quiet --wait --norestart --nocache modify `
        --installPath "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools" `
        --add Microsoft.VisualStudio.Workload.AzureBuildTools `
        --remove Microsoft.VisualStudio.Component.Windows10SDK.10240 `
        --remove Microsoft.VisualStudio.Component.Windows10SDK.10586 `
        --remove Microsoft.VisualStudio.Component.Windows10SDK.14393 `
        --remove Microsoft.VisualStudio.Component.Windows81SDK `
        || IF "%ERRORLEVEL%"=="3010" EXIT 0) `
    `
    # Cleanup
    && del /q vs_buildtools.exe

RUN mkdir C:\DOWNLOADS

RUN curl -SL --output cmake-3.22.0-rc1-windows-x86_64.zip https://github.com/Kitware/CMake/releases/download/v3.22.0-rc1/cmake-3.22.0-rc1-windows-x86_64.zip && `
    unzip

RUN Invoke-WebRequest https://github.com/microsoft/LightGBM/releases/download/v3.3.0/LightGBM-complete_source_code_zip.zip -outfile C:\DOWNLOADS\LightGBM-complete_source_code_zip.zip && `
    Expand-Archive -Path C:\DOWNLOADS\LightGBM-complete_source_code_zip.zip -DestinationPath C:\LightGBM

# Define the entry point for the docker container.
# This entry point starts the developer command prompt and launches the PowerShell shell.
ENTRYPOINT ["C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\Common7\\Tools\\VsDevCmd.bat", "&&", "powershell.exe", "-NoLogo", "-ExecutionPolicy", "Bypass"]
