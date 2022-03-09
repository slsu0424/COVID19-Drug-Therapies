Install-Module AzureAD

Connect-AzureAD

Connect-AzureAD -TenantId *Insert Directory ID here*

$appName = "TailwindTradersSalesApp"
$appURI = "https://tailwindtraderssalesapp.twtmitt.onmicrosoft.com"
$appHomePageUrl = "http://www.tailwindtraders.com/"
$appReplyURLs = @($appURI, $appHomePageURL, "https://localhost:1234")
if(!($myApp = Get-AzureADApplication -Filter "DisplayName eq '$($appName)'"  -ErrorAction SilentlyContinue))
{
    $myApp = New-AzureADApplication -DisplayName $appName -IdentifierUris $appURI -Homepage $appHomePageUrl -ReplyUrls $appReplyURLs    
}


#2. Add App Key
$Guid = New-Guid
$startDate = Get-Date
    
$PasswordCredential = New-Object -TypeName Microsoft.Open.AzureAD.Model.PasswordCredential
$PasswordCredential.StartDate = $startDate
$PasswordCredential.EndDate = $startDate.AddYears(1)
$PasswordCredential.KeyId = $Guid
$PasswordCredential.Value = ([System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes(($Guid))))
