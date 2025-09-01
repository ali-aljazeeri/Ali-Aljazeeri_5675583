import { Link, usePage } from "@inertiajs/react";
import FlashMessage from "@/Components/FlashMessage";

export default function AdminLayout({ children, title, description }) {
    const { flash = {} } = usePage().props;

    return (
        <div className="min-h-screen bg-gray-100">
            <header className="bg-white shadow-sm">
                <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between">
                        <h1 className="text-2xl font-bold text-gray-900">
                            {title}
                        </h1>
                        <nav className="flex gap-4">
                            <Link
                                href={route('damages.index')}
                                className={`text-sm font-medium ${route().current('damages.index') ? 'text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}
                            >
                                Damage Reports
                            </Link>
                        </nav>
                    </div>
                    {description && (
                        <p className="mt-1 text-sm text-gray-500">
                            {description}
                        </p>
                    )}
                </div>
            </header>

            <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
                {/* Flash Messages */}
                <div className="space-y-4 mb-6">
                    {flash.success && <FlashMessage type="success" message={flash.success} />}
                    {flash.error && <FlashMessage type="error" message={flash.error} />}
                    {flash.info && <FlashMessage type="info" message={flash.info} />}
                    {flash.warning && <FlashMessage type="warning" message={flash.warning} />}
                </div>

                {children}
            </main>
        </div>
    );
}
