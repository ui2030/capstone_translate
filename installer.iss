; Cocktail Overlay Translator — Inno Setup script
;
; 빌드 절차:
;   1. pip install -r requirements-dev.txt
;   2. python download_tessdata.py    (tessdata/*.traineddata 다운로드)
;   3. python cocktail.py --make-icon  (assets/icon.ico 생성 — 코드로 그린다)
;   4. pyinstaller build.spec          (dist/Cocktail/ 생성)
;   5. Inno Setup Compiler(ISCC.exe)로 이 파일 컴파일 → Output\Cocktail-Setup-1.0.0.exe
;
; traineddata는 컴포넌트로 선택 설치 (eng 필수, 나머지 선택).
; 사용자가 선택한 traineddata만 {app}\tessdata\ 에 설치되며,
; 본 앱은 시작 시 TESSDATA_PREFIX를 자동으로 그쪽으로 설정함 (cocktail_ocr.py).
;
; PK-1: Tesseract 엔진(tesseract.exe + DLL, Apache 2.0)은 build.spec 이 dist 에 동봉하므로
; 사용자가 따로 설치할 것이 **없다**. 라이선스 원문은 {app}\_internal\tesseract\doc\LICENSE.

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
; IC-2: 아이콘 원본은 코드다 (cocktail_ui.make_app_icon → `python cocktail.py --make-icon`).
SetupIconFile=assets\icon.ico
OutputDir=Output
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
; 본체 — PyInstaller 결과물 (Tesseract 엔진 포함).
; _internal\tessdata 는 뺀다 — traineddata 는 아래 컴포넌트로 사용자가 고른 것만 {app}\tessdata 에 깔린다
; (앱은 {app}\tessdata 를 먼저 본다: cocktail_ocr._SEARCH_DIRS).
Source: "dist\Cocktail\*"; DestDir: "{app}"; Excludes: "_internal\tessdata\*"; Flags: recursesubdirs ignoreversion; Components: main

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

; PK-1: Tesseract 엔진은 동봉되므로 별도 설치 안내가 필요 없다 → 빈 InitializeSetup 삭제.
