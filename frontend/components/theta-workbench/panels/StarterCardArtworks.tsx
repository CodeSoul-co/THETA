import type { ReactElement } from 'react'

export const ResearchIdeaArtwork = (): ReactElement => (
  <svg viewBox="0 0 320 200" fill="none" xmlns="http://www.w3.org/2000/svg" focusable="false">
    <defs>
      <linearGradient id="ri-panel" x1="66" y1="34" x2="286" y2="179" gradientUnits="userSpaceOnUse"><stop stopColor="#FFFFFF" /><stop offset="1" stopColor="#F6F9FF" /></linearGradient>
      <linearGradient id="ri-core" x1="151" y1="87" x2="188" y2="124" gradientUnits="userSpaceOnUse"><stop stopColor="#80AEFF" /><stop offset="1" stopColor="#526FE5" /></linearGradient>
      <filter id="ri-back" x="6" y="-4" width="282" height="198" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB"><feDropShadow dx="0" dy="12" stdDeviation="12" floodColor="#48658F" floodOpacity="0.12" /></filter>
      <filter id="ri-front" x="32" y="10" width="288" height="190" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB"><feDropShadow dx="0" dy="14" stdDeviation="12" floodColor="#38537C" floodOpacity="0.17" /></filter>
      <filter id="ri-glow" x="137" y="72" width="68" height="68" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB"><feDropShadow dx="0" dy="5" stdDeviation="7" floodColor="#5B7FE8" floodOpacity="0.27" /></filter>
    </defs>
    <g filter="url(#ri-back)" transform="rotate(-4.5 143 94)">
      <rect x="28" y="19" width="232" height="148" rx="20" fill="#EAF2FF" stroke="#C7D6EC" />
      <path d="M28 39C28 27.954 36.954 19 48 19H240C251.046 19 260 27.954 260 39V52H28V39Z" fill="#FAFCFF" fillOpacity="0.84" />
      <circle cx="47" cy="35" r="3" fill="#AFC3E3" /><circle cx="57" cy="35" r="3" fill="#CBD8EA" /><circle cx="67" cy="35" r="3" fill="#DCE5F1" />
      <rect x="45" y="72" width="54" height="8" rx="4" fill="#C6D6ED" /><rect x="45" y="92" width="146" height="6" rx="3" fill="#D6E1F0" /><rect x="45" y="108" width="121" height="6" rx="3" fill="#D6E1F0" />
    </g>
    <g filter="url(#ri-front)" transform="rotate(1.2 184 110)">
      <rect x="55" y="37" width="245" height="145" rx="20" fill="url(#ri-panel)" stroke="#C9D7EA" />
      <path d="M55 57C55 45.954 63.954 37 75 37H280C291.046 37 300 45.954 300 57V68H55V57Z" fill="#FBFCFF" />
      <circle cx="75" cy="52" r="3" fill="#9EB8E8" /><circle cx="85" cy="52" r="3" fill="#D2DDED" /><rect x="102" y="48" width="58" height="8" rx="4" fill="#DFE7F2" /><rect x="264" y="47" width="20" height="10" rx="5" fill="#E9F1FF" />
      <path d="M169 105C146 105 140 83 117 83M169 105C145 105 140 139 115 139M169 105C194 105 203 82 229 82M169 105C194 105 205 139 232 139" stroke="#CDD9EC" strokeWidth="2" strokeLinecap="round" />
      <rect x="80" y="72" width="54" height="22" rx="8" fill="#F2F6FC" stroke="#CEDCEF" /><circle cx="91" cy="83" r="4" fill="#81A5EB" /><rect x="100" y="79" width="24" height="7" rx="3.5" fill="#CCD9EB" />
      <rect x="79" y="128" width="55" height="22" rx="8" fill="#F5F8FD" stroke="#D4DFEE" /><circle cx="90" cy="139" r="4" fill="#A3B9E4" /><rect x="99" y="135" width="25" height="7" rx="3.5" fill="#D3DEEC" />
      <rect x="214" y="71" width="55" height="22" rx="8" fill="#F2F6FC" stroke="#CEDCEF" /><circle cx="225" cy="82" r="4" fill="#7299EA" /><rect x="234" y="78" width="25" height="7" rx="3.5" fill="#CAD8EB" />
      <rect x="216" y="128" width="55" height="22" rx="8" fill="#F5F8FD" stroke="#D4DFEE" /><circle cx="227" cy="139" r="4" fill="#9CB4E2" /><rect x="236" y="135" width="25" height="7" rx="3.5" fill="#D3DEEC" />
      <g filter="url(#ri-glow)"><circle cx="169" cy="105" r="19" fill="url(#ri-core)" /><path d="M169 94.5L178 99.7V110.3L169 115.5L160 110.3V99.7L169 94.5Z" stroke="white" strokeWidth="1.8" strokeLinejoin="round" /><circle cx="169" cy="105" r="3.2" fill="white" /></g>
    </g>
    <g transform="rotate(4 267 43)"><rect x="237" y="26" width="59" height="32" rx="9" fill="#FFFFFF" stroke="#CBD9EC" /><rect x="246" y="35" width="10" height="14" rx="5" fill="#E7EFFF" /><rect x="262" y="35" width="23" height="5" rx="2.5" fill="#AFC5EC" /><rect x="262" y="44" width="16" height="4" rx="2" fill="#DEE6F1" /></g>
  </svg>
)

export const ExistingDataArtwork = (): ReactElement => (
  <svg viewBox="0 0 320 200" fill="none" xmlns="http://www.w3.org/2000/svg" focusable="false">
    <defs>
      <linearGradient id="ed-panel" x1="93" y1="55" x2="293" y2="180" gradientUnits="userSpaceOnUse"><stop stopColor="#FFFFFF" /><stop offset="1" stopColor="#F7FAFF" /></linearGradient>
      <filter id="ed-back" x="5" y="-5" width="278" height="202" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB"><feDropShadow dx="0" dy="12" stdDeviation="12" floodColor="#49638A" floodOpacity="0.13" /></filter>
      <filter id="ed-front" x="65" y="24" width="254" height="175" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB"><feDropShadow dx="0" dy="14" stdDeviation="12" floodColor="#35517C" floodOpacity="0.18" /></filter>
    </defs>
    <g filter="url(#ed-back)" transform="rotate(-4 140 96)">
      <rect x="27" y="18" width="230" height="151" rx="19" fill="#FFFFFF" stroke="#C8D7EA" />
      <path d="M27 38C27 26.954 35.954 18 47 18H237C248.046 18 257 26.954 257 38V52H27V38Z" fill="#EDF4FF" />
      <circle cx="46" cy="35" r="3" fill="#86A7E4" /><circle cx="56" cy="35" r="3" fill="#C3D2E8" /><rect x="73" y="31" width="58" height="8" rx="4" fill="#C8D8ED" />
      <path d="M45 69H239M45 91H239M45 113H239M45 135H239M87 59V151M145 59V151M203 59V151" stroke="#E1E8F2" />
      <rect x="49" y="64" width="27" height="7" rx="3.5" fill="#7399E8" /><rect x="94" y="65" width="37" height="5" rx="2.5" fill="#CFDBEB" /><rect x="151" y="65" width="35" height="5" rx="2.5" fill="#D9E2EF" />
      <rect x="49" y="82" width="22" height="5" rx="2.5" fill="#CCD9EB" /><rect x="94" y="82" width="42" height="5" rx="2.5" fill="#D9E2EF" /><rect x="151" y="82" width="28" height="5" rx="2.5" fill="#D9E2EF" />
      <rect x="49" y="104" width="31" height="5" rx="2.5" fill="#CCD9EB" /><rect x="94" y="104" width="34" height="5" rx="2.5" fill="#D9E2EF" /><rect x="151" y="104" width="40" height="5" rx="2.5" fill="#D9E2EF" />
    </g>
    <g filter="url(#ed-front)" transform="rotate(2.5 198 122)">
      <rect x="88" y="49" width="211" height="132" rx="20" fill="url(#ed-panel)" stroke="#C8D6E9" />
      <path d="M88 69C88 57.954 96.954 49 108 49H279C290.046 49 299 57.954 299 69V79H88V69Z" fill="#FBFCFF" />
      <circle cx="107" cy="64" r="3" fill="#9CB7E6" /><circle cx="117" cy="64" r="3" fill="#D3DEED" /><rect x="134" y="60" width="50" height="8" rx="4" fill="#DCE5F1" /><rect x="263" y="59" width="20" height="10" rx="5" fill="#E8F0FF" />
      <path d="M113 153H272M113 153V96" stroke="#D9E3F0" strokeWidth="1.5" strokeLinecap="round" />
      <rect x="127" y="125" width="14" height="28" rx="5" fill="#A6BDEB" /><rect x="151" y="111" width="14" height="42" rx="5" fill="#86A7ED" /><rect x="175" y="93" width="14" height="60" rx="5" fill="#5C84E7" /><rect x="199" y="118" width="14" height="35" rx="5" fill="#99B3E9" />
      <path d="M226 132C238 117 249 124 258 104C264 91 271 96 278 88" stroke="#5E86E9" strokeWidth="3" strokeLinecap="round" />
      <circle cx="226" cy="132" r="3.5" fill="#FFFFFF" stroke="#5E86E9" strokeWidth="2" /><circle cx="258" cy="104" r="3.5" fill="#FFFFFF" stroke="#5E86E9" strokeWidth="2" /><circle cx="278" cy="88" r="3.5" fill="#5E86E9" />
      <rect x="113" y="88" width="42" height="7" rx="3.5" fill="#CEDBED" />
    </g>
    <g transform="rotate(-3 73 154)"><rect x="37" y="132" width="75" height="42" rx="11" fill="#FFFFFF" stroke="#CBD9EB" /><circle cx="55" cy="153" r="8" fill="#E9F1FF" /><path d="M51 153H59M55 149V157" stroke="#678DE8" strokeWidth="2" strokeLinecap="round" /><rect x="70" y="146" width="29" height="6" rx="3" fill="#B9CBE9" /><rect x="70" y="157" width="21" height="5" rx="2.5" fill="#E0E7F1" /></g>
  </svg>
)

export const GuidedStartArtwork = (): ReactElement => (
  <svg viewBox="0 0 320 200" fill="none" xmlns="http://www.w3.org/2000/svg" focusable="false">
    <defs>
      <linearGradient id="gs-panel" x1="57" y1="38" x2="296" y2="179" gradientUnits="userSpaceOnUse"><stop stopColor="#FFFFFF" /><stop offset="1" stopColor="#F6F9FF" /></linearGradient>
      <linearGradient id="gs-start" x1="77" y1="87" x2="112" y2="122" gradientUnits="userSpaceOnUse"><stop stopColor="#7EACFF" /><stop offset="1" stopColor="#4D70E6" /></linearGradient>
      <filter id="gs-back" x="1" y="-3" width="283" height="200" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB"><feDropShadow dx="0" dy="12" stdDeviation="12" floodColor="#48638A" floodOpacity="0.12" /></filter>
      <filter id="gs-front" x="28" y="11" width="291" height="189" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB"><feDropShadow dx="0" dy="14" stdDeviation="12" floodColor="#35517A" floodOpacity="0.18" /></filter>
      <filter id="gs-glow" x="65" y="74" width="62" height="62" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB"><feDropShadow dx="0" dy="5" stdDeviation="7" floodColor="#547BEA" floodOpacity="0.3" /></filter>
    </defs>
    <g filter="url(#gs-back)" transform="rotate(-4.5 139 97)">
      <rect x="24" y="20" width="235" height="150" rx="20" fill="#EAF2FF" stroke="#C8D7EA" />
      <path d="M24 40C24 28.954 32.954 20 44 20H239C250.046 20 259 28.954 259 40V53H24V40Z" fill="#FAFCFF" fillOpacity="0.86" />
      <circle cx="43" cy="36" r="3" fill="#AFC3E3" /><circle cx="53" cy="36" r="3" fill="#CFDBEB" /><circle cx="63" cy="36" r="3" fill="#DCE5F0" />
      <circle cx="74" cy="91" r="9" fill="#FFFFFF" stroke="#9AB5E6" strokeWidth="2" /><path d="M84 91H210" stroke="#D3DEED" strokeWidth="2" strokeLinecap="round" /><circle cx="124" cy="91" r="5" fill="#B4C7E9" /><circle cx="168" cy="91" r="5" fill="#9CB5E6" /><circle cx="211" cy="91" r="7" fill="#7197E8" />
    </g>
    <g filter="url(#gs-front)" transform="rotate(1.2 180 111)">
      <rect x="51" y="38" width="249" height="145" rx="20" fill="url(#gs-panel)" stroke="#C8D6E9" />
      <path d="M51 58C51 46.954 59.954 38 71 38H280C291.046 38 300 46.954 300 58V69H51V58Z" fill="#FBFCFF" />
      <circle cx="70" cy="53" r="3" fill="#9EB8E7" /><circle cx="80" cy="53" r="3" fill="#D1DDEC" /><rect x="97" y="49" width="54" height="8" rx="4" fill="#DDE6F1" /><rect x="264" y="48" width="20" height="10" rx="5" fill="#E8F0FF" />
      <path d="M110 105C136 105 143 84 164 84H184M110 105C140 105 144 105 169 105H184M110 105C136 105 143 137 164 137H184" stroke="#C8D7EC" strokeWidth="2" strokeLinecap="round" />
      <g filter="url(#gs-glow)"><circle cx="96" cy="105" r="17" fill="url(#gs-start)" /><circle cx="96" cy="105" r="7" stroke="white" strokeWidth="2" /><circle cx="96" cy="105" r="2.5" fill="white" /></g>
      <rect x="184" y="72" width="84" height="25" rx="9" fill="#F3F7FD" stroke="#CCD9EC" /><circle cx="198" cy="84.5" r="4.5" fill="#7499E9" /><rect x="209" y="80.5" width="45" height="7" rx="3.5" fill="#C6D5EA" />
      <rect x="184" y="94" width="92" height="25" rx="9" fill="#EDF3FF" stroke="#BFD0EC" /><circle cx="198" cy="106.5" r="4.5" fill="#5D86E9" /><rect x="209" y="102.5" width="53" height="7" rx="3.5" fill="#AFC5EA" />
      <rect x="184" y="126" width="78" height="25" rx="9" fill="#F5F8FD" stroke="#D3DEED" /><circle cx="198" cy="138.5" r="4.5" fill="#9EB6E3" /><rect x="209" y="134.5" width="39" height="7" rx="3.5" fill="#D1DCEB" />
    </g>
    <g transform="rotate(4 258 41)"><rect x="230" y="22" width="60" height="39" rx="11" fill="#FFFFFF" stroke="#CBD9EB" /><circle cx="250" cy="41" r="10" fill="#EAF1FF" /><path d="M250 34.5L254 41L250 47.5L246 41L250 34.5Z" fill="#638AE9" /><circle cx="250" cy="41" r="2.2" fill="white" /><rect x="267" y="35" width="13" height="5" rx="2.5" fill="#B6C9EA" /><rect x="267" y="44" width="10" height="4" rx="2" fill="#E0E7F1" /></g>
  </svg>
)
