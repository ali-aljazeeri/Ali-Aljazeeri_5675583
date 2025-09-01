import { Button } from "@/Components/ui/button";
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/Components/ui/card";
import { CheckCircle2 } from "lucide-react";
import { Link } from "@inertiajs/react";

export default function Success() {
    return (
        <div className="min-h-screen bg-gray-100">
            {/* Header */}
            <header className="bg-white shadow-sm">
                <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
                    <h1 className="text-2xl font-bold text-gray-900">
                        Car Damage Report System
                    </h1>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                <Card className="max-w-md mx-auto">
                    <CardContent className="pt-6 flex flex-col items-center text-center space-y-4">
                        <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center">
                            <CheckCircle2 className="w-10 h-10 text-green-600" />
                        </div>
                        <div className="space-y-2">
                            <h2 className="text-2xl font-semibold text-gray-900">
                                Report Submitted Successfully
                            </h2>
                            <p className="text-gray-500">
                                Your crash report has been received. We will process it and contact you via email if needed.
                            </p>
                        </div>
                        <Link href={route('home')}>
                            <Button className="mt-4">
                                Submit Another Report
                            </Button>
                        </Link>
                    </CardContent>
                </Card>
            </main>
        </div>
    );
}
