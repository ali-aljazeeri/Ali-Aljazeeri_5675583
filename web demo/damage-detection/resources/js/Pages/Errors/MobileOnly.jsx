import { AlertCircle } from "lucide-react";
import { Card, CardContent } from "@/Components/ui/card";

export default function MobileOnly({ message }) {
    return (
        <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
            <Card className="max-w-md w-full">
                <CardContent className="pt-6">
                    <div className="flex items-center gap-4 text-destructive">
                        <AlertCircle className="h-10 w-10" />
                        <div className="space-y-2">
                            <h2 className="text-lg font-semibold">Desktop Access Not Supported</h2>
                            <p className="text-sm">
                                {message || 'This application requires a mobile device with a camera. Please open this page on your smartphone/tablet.'}
                            </p>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
