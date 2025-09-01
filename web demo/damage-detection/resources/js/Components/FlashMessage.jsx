import { Alert, AlertDescription, AlertTitle } from "@/Components/ui/alert";
import { AlertCircle, CheckCircle2, Info, AlertTriangle } from "lucide-react";

export default function FlashMessage({ type, message }) {
    if (!message) return null;

    const icons = {
        success: <CheckCircle2 className="h-4 w-4" />,
        error: <AlertCircle className="h-4 w-4" />,
        info: <Info className="h-4 w-4" />,
        warning: <AlertTriangle className="h-4 w-4" />
    };

    const titles = {
        success: "Success",
        error: "Error",
        info: "Information",
        warning: "Warning"
    };

    const variants = {
        success: "success",
        error: "destructive",
        info: "default",
        warning: "default"
    };

    return (
        <Alert variant={variants[type]}>
            {icons[type]}
            <AlertTitle>{titles[type]}</AlertTitle>
            <AlertDescription>{message}</AlertDescription>
        </Alert>
    );
} 