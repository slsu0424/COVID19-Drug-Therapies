# login azure
Write-Host "Logging into Azure..."
    az Login

# variables
$keyVaultName = "asakeyabcfelaqpgsfnxcy"
$keyVaultSQLUserSecretName = "test01"
$resourceGroupName = "covid2"
$appName = "covid0"

# pass in arguments
$subscriptionId = Read-Host "subscription Id"
$resourcegroupName = Read-Host "resource group name"

Write-Host "Step 1 - Get Azure IDs for the current subscription..."

    echo "Subscription ID:" $subscriptionId

Write-Host "Step 2 - Create App Registration..."

    # create Application Registration object
    $appReg = az ad app create --display-name $appName

    echo $appReg

    # grab JSON objects
    $objectid = (($appReg) | ConvertFrom-JSON).objectId
    $clientid = (($appReg) | ConvertFrom-JSON).appId
    #$displayname = (($appReg) | ConvertFrom-JSON).displayName

    #echo "Display Name:" $displayname
    echo "Object ID:" $objectid
    echo "Application (client) ID:" $clientid

Write-Host "Step 3 - Generate secret..."

    # generate secret for the App
    $secretValue = ((az ad app credential reset --id $clientid --credential-description Secret01) | ConvertFrom-JSON).password

    echo "Secret Value:" $secretValue

    # convert to secure string
    $secureSecret = ConvertTo-SecureString -String $secretValue -AsPlainText -Force

Write-Host "Step 4 - Create Service Principal..."

    # create Azure AAD service principal with name = Application (client) ID
    az ad sp create-for-rbac --name $clientid

    $spid = ((az ad sp list --display-name $clientid) | ConvertFrom-JSON).appId

    echo "Service Principal Name:" $clientid
    echo "Service Principal ID:" $spid

Write-Host "Step 5 - Set Key Vault Access Policy..."

    # set permissions for the service principal
    az keyvault set-policy --name $keyVaultName --secret-permissions set delete get list --object-id $spid

Write-Host "Step 6 - Register Secret in Key Vault..."

    az keyvault secret set --name $keyVaultSQLUserSecretName --vault-name $keyVaultName --value $secureSecret