; Cocktail Overlay Translator — Inno Setup script
;
; 빌드 절차:
;   1. pip install -r requirements-dev.txt
;   2. python download_tessdata.py    (tessdata/*.traineddata 다운로드)
;   3. pyinstaller build.spec          (dist/Cocktail/ 생성)
;   4. Inno Setup Compiler로 이 파일 컴파일 → Cocktail-Setup.exe
;
; traineddata는 컴포넌트로 선택 설치 (eng 필수, 나머지 선택).
; 사용자가 선택한 traineddata만 {app}\tessdata\ 에 설치되며,
; 본 앱은 시작 시 TESSDATA_PREFIX를 자동으로 그쪽으로 설정함 (Cocktail완성본.py).

#define AppName        "Cocktail Overlay Translator"
#define AppVersion     "1.0.0"
#define AppPublisher   "Cocktail"
#define AppExeName     "Cocktail.exe"

[Setup]
AppId={{B7C5E2A0-3E12-4E5A-9B11-COCKTAIL2026}}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\Cocktail
DefaultGroupName=Cocktail
OutputBaseFilename=Cocktail-Setup-{#AppVersion}
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExeName}
DisableProgramGroupPage=auto
PrivilegesRequired=admin

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Components]
Name: "main";          Description: "Cocktail Translator (필수)";                         Types: full compact custom; Flags: fixed
Name: "tess";          Description: "Tesseract 언어 데이터 (OCR)";                        Types: full compact custom; Flags: fixed
Name: "tess\eng";      Description: "영어 (English)";                                       Types: full compact custom; Flags: fixed
Name: "tess\kor";      Description: "한국어 (Korean)";                                      Types: full
Name: "tess\jpn";      Description: "일본어 (Japanese)";                                    Types: full
Name: "tess\chi_sim";  Description: "중국어 간체 (Chinese Simplified)";                     Types: full
Name: "tess\chi_tra";  Description: "중국어 번체 (Chinese Traditional)";                    Types: full
Name: "tess\fra";      Description: "프랑스어 (French)";                                    Types: full
Name: "tess\deu";      Description: "독일어 (German)";                                      Types: full
Name: "tess\rus";      Description: "러시아어 (Russian)";                                   Types: full
Name: "tess\ara";      Description: "아랍어 (Arabic)";                                      Types: full

[Files]
; 본체 — PyInstaller 결과물
Source: "dist\Cocktail\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion; Components: main

; traineddata — 사용자가 선택한 언어만 설치.
; skipifsourcedoesntexist: download_tessdata.py가 일부 파일을 못 받아도 빌드 실패 안 함.
; eng는 필수 — Source 누락 시 빌드 명시적 실패가 더 안전.
; osd는 동봉하지 않는다 (OSD script 판별은 SL-1에서 퇴역).
Source: "tessdata\eng.traineddata";     DestDir: "{app}\tessdata"; Components: tess\eng
Source: "tessdata\kor.traineddata";     DestDir: "{app}\tessdata"; Components: tess\kor; Flags: skipifsourcedoesntexist
Source: "tessdata\jpn.traineddata";     DestDir: "{app}\tessdata"; Components: tess\jpn; Flags: skipifsourcedoesntexist
Source: "tessdata\chi_sim.traineddata"; DestDir: "{app}\tessdata"; Components: tess\chi_sim; Flags: skipifsourcedoesntexist
Source: "tessdata\chi_tra.traineddata"; DestDir: "{app}\tessdata"; Components: tess\chi_tra; Flags: skipifsourcedoesntexist
Source: "tessdata\fra.traineddata";     DestDir: "{app}\tessdata"; Components: tess\fra; Flags: skipifsourcedoesntexist
Source: "tessdata\deu.traineddata";     DestDir: "{app}\tessdata"; Components: tess\deu; Flags: skipifsourcedoesntexist
Source: "tessdata\rus.traineddata";     DestDir: "{app}\tessdata"; Components: tess\rus; Flags: skipifsourcedoesntexist
Source: "tessdata\ara.traineddata";     DestDir: "{app}\tessdata"; Components: tess\ara; Flags: skipifsourcedoesntexist

[Icons]
Name: "{group}\Cocktail Translator"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Cocktail Translator"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,Cocktail Translator}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\tessdata"

[Code]
function InitializeSetup(): Boolean;
begin
  // Tesseract 엔진 자체는 사용자가 별도 설치 필요. 안내만 표시 (필수 아님 — 우리 앱 코드가 _resolve_tesseract로 자동 탐색).
  Result := True;
end;
