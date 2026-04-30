# 코드 서명 가이드 (Phase 6, Windows)

> 사용자 PC에서 인스톨러를 실행할 때 SmartScreen 경고("알 수 없는 게시자")를
> 회피하려면 **Authenticode 코드 서명 인증서**가 필요합니다.

---

## 1. 인증서 종류

| 종류 | 비용 | SmartScreen | 설치 시 표시 |
|---|---|---|---|
| **Standard Code Signing** | ~$200~400/년 | 평판 누적 후 통과 (수개월) | 게시자 이름 표시 |
| **EV (Extended Validation) Code Signing** | ~$400~700/년 | 즉시 통과 ⭐ | 게시자 이름 표시 |
| 자체 서명 (self-signed) | 무료 | ❌ 사용자가 매번 경고 | "알 수 없는 게시자" |

**추천**: EV Code Signing (인스턴트 신뢰). 단, 하드웨어 토큰(USB)으로만 발급되어 CI 자동 서명이 까다로움.

---

## 2. 발급 기관 (CA)

- **Sectigo** (구 Comodo) — 가장 저렴
- **DigiCert** — 가장 신뢰, 가격 높음
- **GlobalSign**
- **SSL.com**

신원 검증(D-U-N-S 번호, 사업자 등록증, 전화 인터뷰 등)에 1주~1개월 소요.

---

## 3. 서명 절차 (signtool)

Windows SDK의 `signtool.exe` 사용.

### .exe 서명
```cmd
signtool sign /tr http://timestamp.sectigo.com /td sha256 /fd sha256 ^
  /n "Cocktail Translator Project" ^
  "dist\Cocktail\Cocktail.exe"
```

### 인스톨러 서명
```cmd
signtool sign /tr http://timestamp.sectigo.com /td sha256 /fd sha256 ^
  /n "Cocktail Translator Project" ^
  "Output\Cocktail-Setup-1.0.0.exe"
```

`/tr`: 타임스탬프 서버 (인증서 만료 후에도 서명 유효성 유지).
`/n`: 인증서의 Common Name (CN). `/sha1` 으로 thumbprint 사용도 가능.

---

## 4. CI 통합 (GitHub Actions)

### Standard 인증서 (PFX 파일)
- PFX를 base64 인코딩 → GitHub Secrets `WINDOWS_CERT_BASE64`
- 패스워드 → Secrets `WINDOWS_CERT_PASSWORD`

```yaml
- name: Sign EXE
  run: |
    [System.IO.File]::WriteAllBytes("cert.pfx", [Convert]::FromBase64String("${{ secrets.WINDOWS_CERT_BASE64 }}"))
    signtool sign /f cert.pfx /p "${{ secrets.WINDOWS_CERT_PASSWORD }}" `
      /tr http://timestamp.sectigo.com /td sha256 /fd sha256 `
      "dist\Cocktail\Cocktail.exe"
    Remove-Item cert.pfx
  shell: pwsh
```

### EV 인증서 (HSM 필요)
GitHub Actions 무료 러너에선 어려움. AWS CloudHSM 또는 Azure Key Vault에 EV 인증서를 두고 원격 서명 — 비용/세팅 큼. 보통 로컬 보안 머신에서 수동 서명.

---

## 5. 서명 검증

### Windows 탐색기
파일 우클릭 → 속성 → 디지털 서명 탭. "확인됨" 표시 + 게시자 이름 + 타임스탬프 확인.

### CLI
```cmd
signtool verify /pa /v "dist\Cocktail\Cocktail.exe"
```

---

## 6. SmartScreen 평판 (Standard 인증서)

Standard 인증서는 SmartScreen 평판이 0부터 시작. 사용자 다운로드/실행 누적이 임계치(약 수천 회) 넘으면 경고 사라짐.
EV는 즉시. 초기 배포 한 번이라도 중요하면 EV 권장.

---

## 7. 본 프로젝트 단계

현재 **단계 6** 진입 전 준비 자료. 인증서 확보는 사용자/조직 결정 사항.

체크리스트:
- [ ] 사업자 등록 / 개인 사업자 / 법인 결정
- [ ] D-U-N-S 번호 (법인 발급 시)
- [ ] CA 선정 → 인증서 구매
- [ ] 신원 검증 완료
- [ ] 빌드 파이프라인에 signtool 통합 (`installer.iss` 빌드 후)
- [ ] 첫 배포 → SmartScreen 평판 누적 모니터링

---

## 참고

- Microsoft 공식: https://learn.microsoft.com/en-us/windows/win32/seccrypto/cryptography-tools
- signtool: https://learn.microsoft.com/en-us/windows/win32/seccrypto/signtool
- SmartScreen: https://learn.microsoft.com/en-us/windows/security/threat-protection/microsoft-defender-smartscreen/microsoft-defender-smartscreen-overview
