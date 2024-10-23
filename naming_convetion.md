## Audio samples naming conversion

In general, the name for sample should contain following information:

- Source of sample - which audio was initially used <#1>
- Transformations and distortions applied to audio, such as distortions, filters, etc. <#2>
- Watermark, namely technique and message used <#3>
- Voice cloning technique used (if applied) <#4>

Possible additions: sampling rate, language?

Then the structure of the name is
<#1>\_<#2>\_<#3>\_<#4>.\<extension\>

Here is additional specification:

<#1> - should refer to filename of the source audio sample which was at the input of pipeline, allowed characters: alphanumeric + "-"

<#2> - each transformation should have its' own code:

- "1" - conversion from stereo to mono
- "2" - TBD

<#3> - each watermark technique should have its' own code:
- "1" - wavmark
- "2" - AudioSeal
- "3" - SilentCipher
- "4" - audiowmark

<#4> - each Voice cloning technique should have its' own code:
- "1" - OpenVoice V2
- "2" - TBD